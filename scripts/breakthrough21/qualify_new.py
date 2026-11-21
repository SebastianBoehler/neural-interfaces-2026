"""Check the exact signed candidate's terminal path or disabled prior on public windows."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd() / "submissions/final_stack"))
import check_hard_team_b as C

LEAD = "final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip"
LEAD_SHA = "7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07"


def unpack(zip_path, sha, target, data, device, *, zero=False):
    C.verify_parent(zip_path, sha)
    module = C.unpack(zip_path, target)
    terminal = getattr(module, "BreakthroughStack", None)
    with ExitStack() as hooks:
        if zero and terminal is not None:
            for key, value in dict(MIXES={}, SPECTRAL_SHA=None, CB_FILES=()).items():
                hooks.enter_context(patch.object(terminal, key, value))
        model = C.load(module, target, data, device)
        if zero and terminal is None:
            if getattr(model.team_b, "population_cov", None) is None:
                raise ValueError("candidate has no prior or terminal arm")
            model.team_b.population_cov = None
    C.clean(model)
    if zero and (getattr(model, "spectral", None) is not None or getattr(model, "terminal_cb", None) is not None):
        raise ValueError("disabled terminal candidate still loaded an expert")
    return model


def check_active(model, windows):
    native = C.fresh_model(model)
    label, changes = [], []
    terminal = getattr(native, "terminal_scores", None)

    def tap(q, fused):
        result = terminal(q, fused)
        normalized = np.exp(q - q.max()) if fused else q
        normalized = normalized / normalized.sum()
        changes.append(float(np.max(np.abs(result - normalized))))
        return result

    with ExitStack() as hooks:
        if terminal is not None:
            hooks.enter_context(patch.object(native, "terminal_scores", tap))
        for x in windows:
            label.append(C.label(native, x))
    if native.dead or native.log6["err"] or native.log6["v6"] != len(windows):
        raise ValueError(f"official Solver degraded: {native.log6}")
    if native.cf.dead or native.cp.dead or native.cf.log["err"] or native.cp.log["err"]:
        raise ValueError("terminal stage failed")
    sb, cb = getattr(native, "spectral", None), getattr(native, "terminal_cb", None)
    if terminal is not None:
        if not changes or max(changes) < 1e-4 or (sb is None and cb is None):
            raise ValueError("terminal expert was inactive")
        if sb is not None and sb.n != len(windows):
            raise ValueError("spectral expert missed a window")
        if cb is not None and (cb.n != len(windows) or cb._forward_graph.replays != len(windows)):
            raise ValueError("CBraMod graph missed a window")
    elif native.team_b._lookup_prior(1) != 12:
        raise ValueError("cold-start covariance was inactive")
    C.check_features(native.team_b, len(windows))
    C.clean(model)
    return np.asarray(label), (max(changes) if changes else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "zero"))
    p.add_argument("candidate", type=Path)
    p.add_argument("sha")
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    C.verify_parent(a.data, C.DATA_SHA)
    data = C.D.load(a.data)
    device = "cuda:0" if a.stage == "smoke" else "cpu"
    report = {}
    rows_map = C.D.plans(data, 71120, C.ZERO_STREAMS)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        lead = unpack(a.candidate.parent / LEAD, LEAD_SHA, root / "lead", data, device)
        trial = unpack(a.candidate, a.sha, root / "trial", data, device, zero=a.stage == "zero")
        for session, r in rows_map.items():
            rows = r[:72] if a.stage == "smoke" else r
            x = torch.from_numpy(data["X"][rows]).to(device)
            ref = C.stream(lead, x, capture=True)
            if a.stage == "zero":
                vals = C.stream(trial, x, capture=True)
                for A, B in zip(ref[:2], vals[:2], strict=True):
                    if A.tobytes() != B.tobytes():
                        raise ValueError(f"zero labels or upstream taps changed at {session}")
                report[session] = dict(windows=len(x), changed=0, bitwise=True)
            else:
                label, movement = check_active(trial, x)
                repeat, _ = check_active(trial, x[:32])
                np.testing.assert_array_equal(label[:32], repeat)
                report[session] = dict(windows=len(x), changed=int((label != ref[0]).sum()),
                                       max_terminal_posterior_movement=movement)
    n = sum(row["windows"] for row in report.values())
    if a.stage == "zero" and n != 379:
        raise ValueError("379-window CPU zero check incomplete")
    a.out.write_text(json.dumps(dict(windows=n, sessions=report, sha=a.sha), indent=1) + "\n")
    print(f"PASS {a.stage} {n} windows", flush=True)


if __name__ == "__main__":
    main()
