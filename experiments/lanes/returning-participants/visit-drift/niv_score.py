"""Lane D3 NI-V scoring (local copy of the Galvani out/ dirs). usage: niv_score.py OUT_DIR WINDOWS.npz DEST.json
fa  : package k (person k's visit 1 enrolled), streams of the 8 other people, both visits, order full:0 = wrong-enrolment harm check.
ret : person k visit 2 on package k, 16 orders (B-ret).
BA3 per stream; delta SYS - hb paired per stream; 80 % intervals resample the streaming people (9). Labels only; y read here for scoring."""
import json
import sys
from pathlib import Path

import numpy as np

out, wpath, dest = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with np.load(wpath, allow_pickle=False) as z:
    Y, SUBJ = z["y"], z["subject"]
names = sorted(set(SUBJ))
rng = np.random.default_rng(3030)
ba = lambda y, p: 100 * np.mean([(p[y == c] == c).mean() for c in range(3)])


def read(mode, sys_):
    """{(k, tag): (ba, rec, routed)} ; tag = scheme|idx|subject|visit"""
    r = {}
    for k in range(9):
        for f in (out / f"{mode}_{sys_}_k{k}").glob("p*_c*.npz"):
            with np.load(f) as z:
                for t in {t.rsplit("|", 1)[0] for t in z.files}:
                    rows, lab, meta = z[t + "|rows"], z[t + "|labels"], z[t + "|meta"]
                    r[(k, t)] = (ba(Y[rows], lab), meta[0], meta[1])
    return r


def boot(x):
    x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
    return [float(x.mean()), float(np.percentile(b, 10)), float(np.percentile(b, 90)), len(x)]


res = {}
for mode in ("fa", "ret"):
    ref = read(mode, "hb")
    for s in ("D3f", "D3g"):
        cur = read(mode, s)
        keys = sorted(set(ref) & set(cur))
        if not keys:
            continue
        person = lambda k, t: t.split("|")[2] if mode == "fa" else names[k]
        d = {key: cur[key][0] - ref[key][0] for key in keys}
        by = {}
        for key, v in d.items():
            by.setdefault(person(*key), []).append(v)
        pp = {p: float(np.mean(v)) for p, v in by.items()}
        routed = [d[key] for key in keys if ref[key][2] > 0]
        res[f"{mode}_{s}"] = dict(streams=len(keys), routed_streams=len(routed), changed=int(sum(v != 0 for v in d.values())),
                                  delta_people=boot(list(pp.values())), delta_routed_streams=boot(routed) if routed else None,
                                  worst_person=float(min(pp.values())), worst_stream=float(min(d.values())),
                                  hb_abs=float(np.mean([ref[k][0] for k in keys])), recog_equal=bool(all(cur[k][1] == ref[k][1] for k in keys)))
Path(dest).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
