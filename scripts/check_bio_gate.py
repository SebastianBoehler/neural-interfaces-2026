"""Integration check of a routing-v4 package against its base package on all public windows.

Usage: python scripts/check_bio_gate.py PKG_V4 PKG_BASE [--out JSON]
Checks: (1) the gate decisions recorded inside the package equal a stand-alone bio_gate.Bio fed the same windows;
(2) in every session the final posteriors equal the base package's up to the first window with a gate ramp > 0
(the gate cannot change anything earlier); (3) reports per session the first routed window, the person and scores.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402
from bio_gate import Bio  # noqa: E402
from check_compat import post_pass  # noqa: E402


def main():
    v4, base = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); rows = np.ones(len(d["y"]), bool)
    m = load_solver(v4, "v4"); gate = []; step0 = m.route.step

    def step(*a):
        o = step0(*a); gate.append((m.route.bio.last or (-1, np.nan, 0.0)) + (m.route.last or (-1, np.nan, 0.0))); return o
    m.route.step = step
    new = post_pass(m, d, rows); old = post_pass(load_solver(base, "base"), d, rows); gate = np.array(gate, float)
    z = dict(np.load(v4 / "weights.npz")); names = [str(n) for n in z["route_names"]]; ref = Bio(z); st = d["stream"]; res, ok1, ok2 = {}, True, True
    acc = lambda lab: float(np.mean(lab == d["y"]))  # noqa: E731
    for s in dict.fromkeys(st):
        r = np.flatnonzero(st == s); ref.reset_state(); alone = np.array([ref.step(x) for x in d["X"][r]], float)
        ok1 &= bool(np.array_equal(alone[:, 0], gate[r, 0]) and np.array_equal(alone[:, 1], gate[r, 2]))
        rb = gate[r, 2]; first = int(np.flatnonzero(rb > 0)[0]) if (rb > 0).any() else len(r)
        same = np.array_equal(new["route"][r[:first]], old["route"][r[:first]]); ok2 &= bool(same)
        res[s.split("/")[-1]] = dict(first_bio_window=first if first < len(r) else None, person=names[int(gate[r[first], 0])] if first < len(r) else None,
                                     max_score=float(np.nanmax(gate[r, 1])), equal_before_first=bool(same),
                                     labels_changed=int((new["labels"][r] != old["labels"][r]).sum()))
    out = dict(gate_equals_standalone=ok1, equal_to_base_before_first_routed_window=ok2, in_sample_acc_v4=acc(new["labels"]), in_sample_acc_base=acc(old["labels"]),
               labels_changed=int((new["labels"] != old["labels"]).sum()), sessions=res)
    print(json.dumps(out, indent=1))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(out, indent=1))
    assert ok1 and ok2


if __name__ == "__main__":
    main()
