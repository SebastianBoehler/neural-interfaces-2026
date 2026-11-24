"""Lane S: seal ensemble banks for arms P/W/D (laneH format) next to Lane H's A/B and score standalone held-new LOPO.

python laneS/collect_s.py ROOT WORK   WORK holds h_{A,B,P,W,D}_{0,1,2}_{P01..P10,all}.{npz,json} (A/B copied from the Lane H run).
Writes WORK/ensemble_{A,B,P,W,D}_held_new.npz and WORK/laneS_standalone.json (D-B, P-B, W-B, D-A, B-A: paired people+orders 80 %)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT0 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT0 / "laneH"), str(ROOT0 / "laneC")]
import collect as H  # noqa: E402
from oracle import metadata, require, sha  # noqa: E402

ARMS = "ABPWD"
PAIRS = (("B", "D"), ("B", "P"), ("B", "W"), ("A", "D"), ("A", "B"), ("P", "D"), ("W", "D"))


def main():
    root, work = Path(sys.argv[1]), Path(sys.argv[2])
    d = metadata(root / "data/proteus_windows.npz")
    require(sha(root / "runs/shuffle_perms.npz") == H.ORDER_SHA, "orders changed")
    rep, banks = {}, {}
    with np.load(root / "runs/shuffle_perms.npz", allow_pickle=False) as perms:
        for arm in ARMS:
            folds = {h: [H.fold(work, arm, s, h, d, perms) for s in H.SEEDS] for h in H.ALL}
            for h in H.ALL:  # the receipt must name this arm's own trunk
                for s in H.SEEDS:
                    r = json.loads((work / f"h_{arm}_{s}_{h}.json").read_text())
                    require(arm == "A" or r["init"].endswith(f"-hs{arm.lower()}-s{{seed}}.pt") or arm == "B" and "-hp7-" in r["init"], f"wrong trunk {r['init']}")
                    require(r["epochs"] == 40 and r["k_test"] == 20.0, f"recipe differs {arm} {s} {h}")
            rows = np.concatenate([v[0][0] for v in folds.values()]); idx = np.argsort(rows)
            np.testing.assert_array_equal(rows[idx], np.arange(len(d["y"])))
            want = dict(index=rows[idx])
            for o in H.ORDERS:
                mem = np.stack([np.concatenate([v[s][1][o] for v in folds.values()])[idx] for s in range(len(H.SEEDS))])
                want[o], want[o + "_members"] = mem.mean(0), mem
                want[o + "_delivery"] = perms[f"full|{o[-1]}|all_loso"]
            np.savez(work / f"ensemble_{arm}_held_new.npz", **want)
            banks[arm] = want
            rep[arm] = dict(sha=sha(work / f"ensemble_{arm}_held_new.npz"),
                            ba_ensemble={o: H.ba(d, want["index"], want[o].argmax(1)) for o in H.ORDERS})
    for a, b in PAIRS:
        per = [H.person_delta(d, banks[a][o].argmax(1), banks[b][o].argmax(1)) for o in H.ORDERS]
        mean, iv = H.interval(per)
        rep[f"{b}-{a}"] = dict(delta_pp=mean, paired_people_orders_80=iv)
    (work / "laneS_standalone.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: (v["ba_ensemble"]["full0"] if k in ARMS else v) for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
