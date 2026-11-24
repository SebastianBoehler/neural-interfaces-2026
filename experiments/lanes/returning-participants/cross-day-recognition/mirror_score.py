"""Test 2 scoring: per-person cell BA and MI recall (class 0), views orig / mirror / mean; seeds averaged as an ensemble, orders full0-3 averaged.
python laneQ/mirror_score.py MIRDIR"""
import sys
import numpy as np
d = sys.argv[1]
z = np.load("data/proteus_windows.npz", allow_pickle=True)
y, sub, ses, itf = z["y"], z["subject"].astype(str), z["session"].astype(str), z["interface"].astype(str)
who = np.array([s.rsplit("/", 1)[-1] for s in sub]); cell = np.array([f"{a}|{b}|{c}" for a, b, c in zip(sub, ses, itf)])
ORD = [f"full{i}" for i in range(4)]
def ba(t, p):
    r = [(p[t == k] == k).mean() for k in range(3) if (t == k).any()]
    return float(np.mean(r))
res = {}
for cfg in ("fbcnet_raw_sbn", "fblight_ea_rot05_sbn"):
    for P in [f"P{i:02}" for i in range(1, 11)]:
        zs = [np.load(f"{d}/q_mirror_{cfg}_{s}_{P}.npz") for s in range(3)]
        q = zs[0]["index"]
        for o in ORD:
            for k in range(3):
                assert (zs[k][f"orig|{o}|delivery"] == zs[0][f"orig|{o}|delivery"]).all() and (zs[k][f"mirror|{o}|delivery"] == zs[0][f"orig|{o}|delivery"]).all()
        for view in ("orig", "mirror", "mean"):
            ba_o, mi_o = [], []
            for o in ORD:
                po = np.mean([k[f"orig|{o}"] for k in zs], 0); pm = np.mean([k[f"mirror|{o}"] for k in zs], 0)
                p = {"orig": po, "mirror": pm, "mean": 0.5 * (po + pm)}[view].argmax(1)
                t = y[q]; c = cell[q]
                ba_o.append(np.mean([ba(t[c == u], p[c == u]) for u in sorted(set(c))]))
                mi_o.append(np.mean([(p[(c == u) & (t == 0)] == 0).mean() for u in sorted(set(c)) if ((c == u) & (t == 0)).any()]))
            res[(cfg, P, view)] = (100 * np.mean(ba_o), 100 * np.mean(mi_o))
for cfg in ("fbcnet_raw_sbn", "fblight_ea_rot05_sbn"):
    print(f"== {cfg}: BA / MI recall (%), cell-equal, orders full0-3, 3-seed ensemble")
    print("person  orig        mirror      mean       d(mean-orig) BA / MI")
    dBA, dMI = [], []
    for P in [f"P{i:02}" for i in range(1, 11)]:
        o, m, a = (res[(cfg, P, v)] for v in ("orig", "mirror", "mean"))
        dBA.append(a[0] - o[0]); dMI.append(a[1] - o[1])
        print(f"{P}   {o[0]:5.1f}/{o[1]:5.1f}  {m[0]:5.1f}/{m[1]:5.1f}  {a[0]:5.1f}/{a[1]:5.1f}   {a[0]-o[0]:+5.2f} / {a[1]-o[1]:+5.2f}")
    allo = np.mean([res[(cfg, P, 'orig')][0] for P in [f"P{i:02}" for i in range(1, 11)]]); allm = np.mean([res[(cfg, P, 'mirror')][0] for P in [f"P{i:02}" for i in range(1, 11)]])
    print(f"person-mean BA orig {allo:.2f} mirror {allm:.2f} mean-view delta {np.mean(dBA):+.2f}; MI-recall delta mean {np.mean(dMI):+.2f}; min person BA delta {min(dBA):+.2f}")
