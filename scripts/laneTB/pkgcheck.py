"""Exact nine25 ZIP checks on the GPU: blend zero = hardcb_safe, blend on moves labels, session isolation, dead guard.

python pkgcheck.py --zip CAND --base HARDCB_SAFE --data NPZ --out JSON
"""
import argparse
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from run import D, DEVICE, stream_session, timed_load
from build import SAFE_SHA, sha
from check_r1a import digest

CAND_SHA = "af40e37507996302e94f2d69df6beec20405cb6932965a5766352d3cb44b9c79"
ORDER = 71124
STREAMS = ["P01|1", "P04|2"]


def unpack(path, out, zero=False):
    with ZipFile(path) as z:
        z.extractall(out)
    if zero:
        sub = out / "submission.py"
        text = sub.read_text()
        assert text.count("    TEAM_B_MIX = 0.25\n") == 1
        sub.write_text(text.replace("    TEAM_B_MIX = 0.25\n", "    TEAM_B_MIX = 0.0\n"))
    return out


def roster(cand):
    with ZipFile(cand) as z:
        pack = torch.load(io.BytesIO(z.read("team_b.pt")), map_location="cpu", weights_only=True)
    return dict(people=pack["people"], first_only=pack["first_only"], ext=pack.get("ext"), steps=pack["steps"],
                nets=[k for k, _ in pack["nets"]], experts=bool(pack.get("experts")))


def dead_guard(template, signals, rows):
    """Force a Team B failure mid-session: the session must finish on the fallback, not crash."""
    official = D.official()[1]
    model = official.fresh_model(template)
    calls = {"n": 0}
    original = model.team_b.posterior

    def failing(window):
        calls["n"] += 1
        if calls["n"] == 40:
            raise ValueError("injected Team B failure")
        return original(window)

    model.team_b.posterior = failing
    labels = []
    try:
        labels = [int(model.predict(signals[r:r + 1].to(DEVICE))[0]) for r in rows]
    except Exception as err:  # recorded, then failed below: the sealed run would crash this session
        return dict(windows=len(labels), raised=repr(err), dead=False, valid=False)
    return dict(windows=len(labels), dead=bool(model.dead), err=str(model.log6["err"]),
                valid=len(labels) == len(rows) and all(0 <= x <= 2 for x in labels))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert digest(a.zip) == CAND_SHA and digest(a.base) == SAFE_SHA
    data = D.load(a.data)
    signals = torch.from_numpy(data["X"])
    plan = D.plans(data, ORDER, STREAMS)
    res = dict(cand_sha=CAND_SHA, base_sha=SAFE_SHA, order=ORDER, team_b_roster=roster(a.zip), streams={})
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        models = {}
        for arm, path, zero in (("base", a.base, False), ("zero", a.zip, True), ("cand", a.zip, False)):
            models[arm], _ = timed_load(unpack(path, tmp / arm, zero), data, DEVICE)
        assert models["base"].__class__.__name__ == "StackEnsemble"
        assert models["zero"].team_b is None and models["zero"].TEAM_B_MIX == 0.0
        assert models["cand"].team_b is not None and models["cand"].TEAM_B_MIX == 0.25
        tapes = {}
        for s, rows in plan.items():
            out = {arm: stream_session(m, signals, rows, DEVICE, capture=True)[:2] for arm, m in models.items()}
            tapes[s] = out["cand"][0]
            res["streams"][s] = dict(
                windows=len(rows),
                zero_vs_base_labels_changed=int(np.count_nonzero(out["zero"][0] != out["base"][0])),
                zero_vs_base_max_posterior=float(np.abs(out["zero"][1] - out["base"][1]).max()),
                cand_vs_base_labels_changed=int(np.count_nonzero(out["cand"][0] != out["base"][0])),
                cand_vs_base_max_posterior=float(np.abs(out["cand"][1] - out["base"][1]).max()))
        # isolation: the second stream alone (fresh template copy) must match the run made after the first stream
        cand = models["cand"]
        assert cand.team_b.windows == [] and cand.team_b.cov is None, "template Team B state was touched"
        alone, _, _ = stream_session(cand, signals, plan[STREAMS[1]], DEVICE)
        res["isolation_labels_changed"] = int(np.count_nonzero(alone != tapes[STREAMS[1]]))
        res["dead_guard"] = dead_guard(cand, signals, plan[STREAMS[0]])
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    s = res["streams"]
    assert all(v["zero_vs_base_labels_changed"] == 0 for v in s.values()), "blend zero changed labels"
    assert all(v["cand_vs_base_labels_changed"] > 0 for v in s.values()), "blend on changed no label"
    assert res["dead_guard"]["dead"] and res["dead_guard"]["valid"], "dead-session guard did not hold"
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
