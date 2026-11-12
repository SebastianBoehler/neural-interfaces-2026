"""Integration check of a v5 package (own-space stage) against its v4.1 base on all public windows.

Usage: python scripts/check_own_stage.py PKG_V5 PKG_BASE [--out JSON]
Checks: (1) in every session the route outputs equal the base package's on every window without a stage;
(2) on staged windows the package's member log posteriors equal the study class (studies/12nov/ownspace.Personal
fitted on all public sessions of the routed person) and the output equals base x exp(beta r log members);
(3) reports per session the first staged window, the person, the labels changed and the in-sample accuracy.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts"), str(ROOT / "studies/12nov")]
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402
from check_compat import post_pass  # noqa: E402
import ownspace as O  # noqa: E402


def main():
    v5, base = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); rows = np.ones(len(d["y"]), bool)
    assert np.array_equal(d["y"], O.y) and np.array_equal(np.asarray(d["X"][::97], np.float64), O.X[::97]), "row order differs from the study data"
    m = load_solver(v5, "v5"); rec = []; step0 = m.route.step

    def step(*a):
        o = step0(*a); rec.append(m.route.stage); return o
    m.route.step = step
    new = post_pass(m, d, rows); old = post_pass(load_solver(base, "base"), d, rows)
    z = dict(np.load(v5 / "weights.npz")); names = [str(n) for n in z["route_names"]]; beta = float(z["own_beta"]); st = d["stream"]
    models, res, ok1, ok2, dmem, dout = {}, {}, True, True, 0.0, 0.0
    for s in dict.fromkeys(st):
        r = np.flatnonzero(st == s); staged = np.array([rec[i] is not None for i in r])
        same = np.array_equal(new["route"][r[~staged]], old["route"][r[~staged]]); ok1 &= bool(same); streams = {}
        for t in np.flatnonzero(staged):
            j, ramp, lm = rec[r[t]]
            if j not in models:
                q = names[j]; models[j] = O.Personal([np.flatnonzero((O.PER == q) & (O.SES == u)) for u in sorted(set(O.SES[O.PER == q]))])
            if j not in streams:
                o = models[j].stream(r); streams[j] = sum(np.log(np.clip(o[k], 1e-3, 1)) for k in "TCS") / 3
            dmem = max(dmem, float(np.abs(streams[j][t] - lm).max()))
            e = np.log(np.clip(old["route"][r[t]], 1e-9, 1)) + beta * ramp * lm; e = np.exp(e - e.max()); dout = max(dout, float(np.abs(e / e.sum() - new["route"][r[t]]).max()))
        first = int(np.flatnonzero(staged)[0]) if staged.any() else None
        res[s.split("/")[-1]] = dict(first_staged_window=first, person=names[rec[r[first]][0]] if first is not None else None, staged_share=float(staged.mean()),
                                     equal_elsewhere=bool(same), labels_changed=int((new["labels"][r] != old["labels"][r]).sum()),
                                     acc_v5=float(np.mean(new["labels"][r] == d["y"][r])), acc_base=float(np.mean(old["labels"][r] == d["y"][r])))
    ok2 = dmem < 1e-8 and dout < 1e-10
    out = dict(equal_to_base_on_unstaged_windows=ok1, members_equal_study_class=bool(dmem < 1e-8), max_member_diff=dmem, max_output_diff=dout,
               in_sample_acc_v5=float(np.mean(new["labels"] == d["y"])), in_sample_acc_base=float(np.mean(old["labels"] == d["y"])),
               labels_changed=int((new["labels"] != old["labels"]).sum()), sessions=res)
    print(json.dumps(out, indent=1))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(out, indent=1))
    assert ok1 and ok2


if __name__ == "__main__":
    main()
