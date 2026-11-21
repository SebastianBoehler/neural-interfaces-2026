"""Check that CUDA graph warm-up leaves no work in the base members' session state."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd() / "submissions/final_stack"))
import check_hard_team_b as C

LEAD = "final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip"
LEAD_SHA = "7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07"


def trace(template, windows, graph=False):
    model = C.fresh_model(template)
    labels = np.asarray([C.label(model, x) for x in windows])
    if model.dead or model.log6["v6"] != len(windows) or model.routeB.log["err"]:
        raise ValueError(f"degraded Solver: {model.log6}")
    if graph:
        bank = model.nets.base_graph
        if bank.graph is None or bank.replays != len(windows) or len(bank.ids) < 24:
            raise ValueError("base CUDA graph did not execute once per window")
        if template.nets.base_graph.graph is not None:
            raise ValueError("base graph escaped its session")
    state = {}
    for _, members in model.nets.groups:
        for i, net in members:
            if not hasattr(net, "m"):
                continue
            for key, value in net.m.named_buffers():
                if key.rsplit(".", 1)[-1] in ("n", "s1", "s2"):
                    state[f"{i}:{key}"] = value.detach().cpu()
    if not state:
        raise ValueError("base SessionBN buffers were absent")
    C.check_features(model.team_b, len(windows))
    return labels, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("sha")
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    C.verify_parent(a.data, C.DATA_SHA)
    C.verify_parent(a.candidate, a.sha)
    lead = a.candidate.parent / LEAD
    C.verify_parent(lead, LEAD_SHA)
    data = C.D.load(a.data)
    counters = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = [C.load(C.unpack(path, root / role), root / role, data, "cuda:0")
                 for role, path in (("lead", lead), ("candidate", a.candidate))]
        for s, rows in C.D.plans(data, 71120, C.ZERO_STREAMS).items():
            x = torch.from_numpy(data["X"][rows[:72]]).cuda()
            expected, buffers = trace(model[0], x)
            actual, state = trace(model[1], x, graph=True)
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(actual[:32], trace(model[1], x[:32], graph=True)[0])
            if buffers.keys() != state.keys():
                raise ValueError("base SessionBN buffer sets differ")
            for key in buffers:
                tol = 0 if key.endswith(".n") else 1e-6
                torch.testing.assert_close(state[key], buffers[key], rtol=tol, atol=5 * tol)
            counters[s] = dict(windows=len(x), checked_buffers=len(state), label_changes=0)
    a.out.write_text(json.dumps(dict(sha=a.sha, sessions=counters), indent=1) + "\n")
    print("PASS labels, SessionBN buffers and graph replay counts", flush=True)


if __name__ == "__main__":
    main()
