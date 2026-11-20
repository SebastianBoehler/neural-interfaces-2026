"""Lane R4b scoring from the small ret.npz files (local). usage: score.py OUT_DIR SET(net|ds7) DEST.json [B10_OUT_DIR]
Per (variant, person, stream, order): BA2 (MI recall, non-MI specificity), recognition, proxy summaries. Deltas vs hardcbB paired per person (people-resampled 80 %, 10 000 draws),
on the orders present in BOTH variants. net: reference variants hardcbB / R4 / R1a / R2 may come from the B10 run (B10_OUT_DIR, tags 'full|72000|2' = scheme|seed|day, 8 orders)."""
import json
import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(2020)
NP = {"net": 19, "ds7": 14}


def streams(f, ref=False):
    """{(scheme, seed, enrol, day): dict(ba2, rec, pxy summary)} of one ret.npz (B10 files: tag scheme|seed|day, enrol 1)"""
    out = {}
    with np.load(f) as z:
        for t in sorted({t.rsplit("|", 1)[0] for t in z.files}):
            sc, sd, ed = t.split("|")
            key = (sc, int(sd), 1, int(ed)) if ref else (sc, int(sd), int(ed[0]), int(ed[1:]))
            y, lab = z[t + "|y"], z[t + "|labels"]
            mi, sp = (lab[y == 1] == 0).mean(), (lab[y == 0] != 0).mean()
            e = dict(ba2=50 * (mi + sp), rec=float(z[t + "|meta"][0]), n=len(y), y=y, lab=lab)
            if t + "|pxy" in z.files:
                e["pxy"] = z[t + "|pxy"]
            out[key] = e
    return out


def load(out, setn, v, ref=False):
    return {k: streams(Path(out) / f"{setn}_{v}_k{k}" / "ret.npz", ref) for k in range(NP[setn]) if (Path(out) / f"{setn}_{v}_k{k}" / "ret.npz").exists()}


def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)])
    return [float(x.mean()), float(np.percentile(b, 10)), float(np.percentile(b, 90)), len(x)]


def paired(A, B):
    """per person mean of (A - B) ba2 over the keys present in both; list over people"""
    d = []
    for k in A:
        if k not in B:
            continue
        ks = sorted(set(A[k]) & set(B[k]))
        if ks:
            d.append(np.mean([A[k][s]["ba2"] - B[k][s]["ba2"] for s in ks]))
    return d


if __name__ == "__main__":
    out, setn, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    b10 = sys.argv[4] if len(sys.argv) > 4 else None
    names = sorted({p.name[len(setn) + 1:].rsplit("_k", 1)[0] for p in Path(out).glob(f"{setn}_*_k*")})
    data = {v: load(out, setn, v) for v in names}
    if b10 and setn == "net":
        for v in ("hardcbB", "hardcbB_R4", "hardcbB_R1a", "hardcbB_R2", "hardcbB_R5", "hardcbB_R3"):
            if v not in data:
                data[v] = {k: streams(Path(b10) / f"{v}_k{k}" / "ret.npz", True) for k in range(19) if (Path(b10) / f"{v}_k{k}" / "ret.npz").exists()}
    res = {"abs": {}, "delta": {}}
    ref = data["hardcbB"]
    for v, D in sorted(data.items()):
        res["abs"][v] = boot([np.mean([s["ba2"] for s in P.values()]) for P in D.values()])
        if v != "hardcbB":
            d = paired(D, ref)
            if d:
                res["delta"][v] = boot(d)
    Path(dest).write_text(json.dumps(res, indent=1))
    for v, e in res["abs"].items():
        dl = res["delta"].get(v)
        print(f"{v:18s} abs {e[0]:.2f} (n {e[3]})" + (f"  delta {dl[0]:+.2f} [{dl[1]:+.2f}, {dl[2]:+.2f}] n {dl[3]}" if dl else ""))
