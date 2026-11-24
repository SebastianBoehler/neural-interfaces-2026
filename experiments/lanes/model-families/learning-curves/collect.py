"""Lane B5: seal one 3-seed ensemble bank per config (laneC format, full0 only) and write per-person standalone BA.

python laneB5/collect.py ROOT WORK OUT    reads WORK/b5_{seed}_{held}.npz (30 files); writes OUT/bank_N{n}s{s}.npz and OUT/standalone.json
Refuses a missing seed/person, a wrong row set or delivery order, or a non-finite posterior."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneC"))
from oracle import ORDER_SHA, metadata, metric_weights, require, sha  # noqa: E402

SEEDS = (0, 1, 2)
PEOPLE = [f"P{i:02}" for i in range(1, 11)]


def main(root, work, out):
    d = metadata(root / "data/proteus_windows.npz")
    require(sha(root / "runs/shuffle_perms.npz") == ORDER_SHA, "orders changed")
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    perm = np.load(root / "runs/shuffle_perms.npz")["full|0|all_loso"]
    folds = {(s, h): np.load(work / f"b5_{s}_{h}.npz") for s in SEEDS for h in PEOPLE}  # KeyError/IOError if partial
    names = [k for k in folds[(0, "P01")].files if k.startswith("N")]
    out.mkdir(exist_ok=True, parents=True)
    res = {}
    for name in names:
        full = np.zeros((len(SEEDS), len(d["y"]), 3))
        for (s, h), z in folds.items():
            q = np.flatnonzero(person == h)
            np.testing.assert_array_equal(z["index"], q)
            np.testing.assert_array_equal(z["delivery"], perm[np.isin(perm, q)])
            p = z[name]
            require(p.shape == (len(q), 3) and np.isfinite(p).all() and (p >= 0).all(), f"bad posterior {name} {s} {h}")
            full[s, q] = p
        require((full.sum(2) > 0.99).all(), f"rows missing in {name}")
        ens = full.mean(0)
        np.savez(out / f"bank_{name}.npz", index=np.arange(len(d["y"])), full0=ens, full0_members=full, full0_delivery=perm)
        pp = {}
        for h in PEOPLE:
            w, cells = metric_weights(d, person == h)
            pp[h] = dict(ens=float(100 * w @ (ens.argmax(1) == d["y"])),
                         single=[float(100 * w @ (full[s].argmax(1) == d["y"])) for s in range(len(SEEDS))], cells=cells)
        w, _ = metric_weights(d, np.ones(len(d["y"]), bool))
        res[name] = dict(sha=sha(out / f"bank_{name}.npz"), people=pp, ba_all=float(100 * w @ (ens.argmax(1) == d["y"])))
    (out / "standalone.json").write_text(json.dumps(res, indent=1) + "\n")
    print({k: round(v["ba_all"], 2) for k, v in res.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
