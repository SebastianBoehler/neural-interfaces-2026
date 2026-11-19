"""Lane V: seal per-arm banks (laneC format) from lopo_v outputs and score standalone held-new LOPO, each arm minus V0.
python laneV/collect_v.py ROOT WORK    -> WORK/ensemble_V*_held_new.npz, WORK/laneV_standalone.json (reuses laneH/collect.py)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneH"))
import collect as C  # noqa: E402
from oracle import metadata, require, sha  # noqa: E402

ARMS = ("V0", "V1", "V2", "V2s", "V3")


def main():
    root, work = Path(sys.argv[1]), Path(sys.argv[2])
    d = metadata(root / "data/proteus_windows.npz")
    banks, rep = {}, {}
    with np.load(root / "runs/shuffle_perms.npz", allow_pickle=False) as perms:
        for arm in ARMS:
            folds = {h: [C.fold(work, arm, s, h, d, perms) for s in C.SEEDS] for h in C.ALL}
            rows = np.concatenate([v[0][0] for v in folds.values()]); idx = np.argsort(rows)
            np.testing.assert_array_equal(rows[idx], np.arange(len(d["y"])))
            want = dict(index=rows[idx])
            for o in C.ORDERS:
                mem = np.stack([np.concatenate([v[s][1][o] for v in folds.values()])[idx] for s in range(len(C.SEEDS))])
                want[o], want[o + "_members"] = mem.mean(0), mem
                want[o + "_delivery"] = perms[f"full|{o[-1]}|all_loso"]
            np.savez(work / f"ensemble_{arm}_held_new.npz", **want)
            banks[arm] = want
            rep[arm] = dict(sha=sha(work / f"ensemble_{arm}_held_new.npz"),
                            ba_ensemble={o: C.ba(d, want["index"], want[o].argmax(1)) for o in C.ORDERS},
                            ba_single_seed_mean={o: float(np.mean([C.ba(d, want["index"], want[o + "_members"][s].argmax(1)) for s in range(3)])) for o in C.ORDERS})
    for b in ARMS[1:]:
        per = [C.person_delta(d, banks["V0"][o].argmax(1), banks[b][o].argmax(1)) for o in C.ORDERS]
        mean, iv = C.interval(per)
        rep[f"{b}-V0"] = dict(delta_pp=mean, paired_people_orders_80=iv, per_person={p: float(np.mean([x[p][0] for x in per])) for p in per[0]})
    (work / "laneV_standalone.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: (v["delta_pp"], v["paired_people_orders_80"]) if "-" in k else float(np.mean(list(v["ba_ensemble"].values()))) for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
