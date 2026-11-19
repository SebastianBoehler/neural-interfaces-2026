"""Lane I: seal ensemble banks per moment rule, nested-LOPO tau (S1), standalone paired contrasts, S0 vs lane H arm A.

python laneI/collect.py ROOT WORK [LANEH_BANK_A]   Writes WORK/ensemble_{S0,S1,S2,k*}_held_new.npz (laneC format) and WORK/laneI_standalone.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT0 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT0 / "laneC"), str(ROOT0 / "laneH")]
from oracle import ORDER_SHA, metadata, require, sha  # noqa: E402
from collect import ba, interval, person_delta  # noqa: E402  (laneH/collect.py)

SEEDS = (0, 1, 2)
ALL = tuple(f"P{i:02}" for i in range(1, 11))
ORDERS = [f"full{i}" for i in range(8)]
TAUS = (5, 10, 20, 40, 80, 160)


def main():
    root, work = Path(sys.argv[1]), Path(sys.argv[2])
    d = metadata(root / "data/proteus_windows.npz")
    require(sha(root / "runs/shuffle_perms.npz") == ORDER_SHA, "orders changed")
    folds = {}
    for h in ALL:
        for s in SEEDS:
            r = json.loads((work / f"i_{s}_{h}.json").read_text())
            require((r["seed"], r["held"]) == (s, h), f"bad receipt {h} {s}")
            folds[h, s] = np.load(work / f"i_{s}_{h}.npz")
    arms = [f"k{t}" for t in TAUS] + ["s2"]
    person = np.array([x.rsplit("/", 1)[-1] for x in d["subject"]])
    n = len(d["y"])
    perms = np.load(root / "runs/shuffle_perms.npz")
    bank = {}
    for a in arms:
        want = dict(index=np.arange(n))
        for o in ORDERS:
            mem = np.zeros((len(SEEDS), n, 3))
            for h in ALL:
                q = np.flatnonzero(person == h)
                for si, s in enumerate(SEEDS):
                    z = folds[h, s]
                    np.testing.assert_array_equal(z["index"], q)
                    ref = perms[f"full|{o[-1]}|all_loso"]
                    np.testing.assert_array_equal(z[o + "_delivery"], ref[np.isin(ref, q)])
                    mem[si, z[o + "_delivery"]] = z[f"{a}|{o}"]
            want[o], want[o + "_delivery"] = mem.mean(0), perms[f"full|{o[-1]}|all_loso"]
        bank[a] = want
    # nested LOPO tau: person P's tau maximises mean standalone BA over the other nine people (orders averaged)
    tba = {t: {h: np.mean([ba(d, np.flatnonzero(person == h), bank[f"k{t}"][o].argmax(1)[np.flatnonzero(person == h)]) for o in ORDERS]) for h in ALL} for t in TAUS}
    chosen = {}
    for h in ALL:
        m = {t: np.mean([tba[t][g] for g in ALL if g != h]) for t in TAUS}
        best = max(m, key=m.get)
        chosen[h] = 20 if m[best] - m[20] < 0.1 else best
    s1 = dict(index=np.arange(n))
    for o in ORDERS:
        p = np.zeros((n, 3))
        for h in ALL:
            q = np.flatnonzero(person == h)
            p[q] = bank[f"k{chosen[h]}"][o][q]
        s1[o], s1[o + "_delivery"] = p, bank["k20"][o + "_delivery"]
    bank["S1"], bank["S0"] = s1, bank["k20"]
    rep = dict(chosen_tau=chosen, ba_by_tau_person_mean={t: float(np.mean(list(tba[t].values()))) for t in TAUS})
    for a in ("S0", "S1", "s2") + tuple(f"k{t}" for t in TAUS if t != 20):
        np.savez(work / f"ensemble_{a}_held_new.npz", **bank[a])
        rep[a] = dict(sha=sha(work / f"ensemble_{a}_held_new.npz"), ba_full0=ba(d, bank[a]["index"], bank[a]["full0"].argmax(1)),
                      ba_orders_mean=float(np.mean([ba(d, bank[a]["index"], bank[a][o].argmax(1)) for o in ORDERS])))
    for a in ("S1", "s2") + tuple(f"k{t}" for t in TAUS if t != 20):
        per = [person_delta(d, bank["S0"][o].argmax(1), bank[a][o].argmax(1)) for o in ORDERS]
        mean, iv = interval(per)
        rep[f"{a}-S0"] = dict(delta_pp=mean, paired_people_orders_80=iv, per_person={p: float(np.mean([x[p][0] for x in per])) for p in per[0]})
    if len(sys.argv) > 3:
        with np.load(sys.argv[3]) as z:
            rep["S0_vs_laneH_A_maxabs_full0"] = float(np.abs(z["full0"] - bank["S0"]["full0"]).max())
            rep["S0_vs_laneH_A_argmax_differ_full0"] = int((z["full0"].argmax(1) != bank["S0"]["full0"].argmax(1)).sum())
    (work / "laneI_standalone.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if "per_person" not in str(v)[:0] and not k.startswith("k")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
