"""Lane K test 4 step 1: label-free individual alpha-peak replication on delivered history halves, genuine vs donor.
python laneK/alpha_peak.py DATA.npz PERMS.npz OUT.json   (registered form in evidence/wave_20nov/laneK_council_tests.md)"""
import json
import sys

import numpy as np
from scipy.signal import welch

FS, AGES, MIN_H = 120, (40, 80), 0.1


def peak(mean_logpsd, f):
    fit = (f >= 3) & (f <= 6) | (f >= 14) & (f <= 30)
    c = np.polyfit(np.log10(f[fit]), mean_logpsd[fit], 1)
    res = mean_logpsd - np.polyval(c, np.log10(f))
    band = np.flatnonzero((f >= 7) & (f <= 13))
    k = band[np.argmax(res[band])]
    if 0 < k < len(f) - 1:  # parabolic refinement
        a, b, d = res[k - 1], res[k], res[k + 1]
        den = a - 2 * b + d
        off = 0.5 * (a - d) / den if den != 0 else 0.0
        return f[k] + off * (f[1] - f[0]), res[k]
    return f[k], res[k]


def main(data, perms, out):
    z = np.load(data, allow_pickle=False)
    names = [str(n) for n in z["ch_names"]]
    post = [i for i, n in enumerate(names) if n.startswith(("P", "O"))]
    assert len(post) == 9, post
    person = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
    perm = np.load(perms)["full|0|all_loso"]
    people = [f"P{i:02}" for i in range(1, 11)]
    X = z["X"]
    psd = {}
    for p in people:
        rows = perm[person[perm] == p]  # delivery order of this person
        f, pw = welch(X[rows][:, post, :].astype(np.float64), fs=FS, nperseg=240, noverlap=120, axis=-1)
        psd[p] = np.log10(pw.mean(1) + 1e-30)  # window x freq (mean over posterior channels of power)
    res = {}
    for split in ("interleaved", "halves"):
        for a in AGES:
            ev = {}
            for p in people:
                idx = np.arange(a)
                A, B = (idx[::2], idx[1::2]) if split == "interleaved" else (idx[: a // 2], idx[a // 2:])
                ev[p] = [peak(psd[p][s].mean(0), f) for s in (A, B)]
            rows = []
            for p in people:
                (pa, ha), (pb, hb) = ev[p]
                gen = abs(pa - pb)
                don = [abs(pa - ev[q][1][0]) for q in people if q != p]
                don_h = [min(ha, ev[q][1][1]) for q in people if q != p]
                rel = min(ha, hb) >= MIN_H
                # donor error over donors that are reliable with the person's half A
                dr = [d for d, h in zip(don, don_h) if h >= MIN_H]
                rows.append(dict(person=p, peak_A=pa, peak_B=pb, hA=ha, hB=hb, genuine=gen, donor=float(np.mean(don)),
                                 donor_rel=float(np.mean(dr)) if dr else None, reliable=bool(rel)))
            ok = [r for r in rows if r["reliable"] and r["donor_rel"] is not None]
            rng = np.random.default_rng(7)
            d = np.array([r["donor_rel"] - r["genuine"] for r in ok]) if ok else np.zeros(1)
            boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
            res[f"{split}_{a}"] = dict(rows=rows, n_reliable=len(ok),
                                       median_genuine=float(np.median([r["genuine"] for r in ok])) if ok else None,
                                       median_donor=float(np.median([r["donor_rel"] for r in ok])) if ok else None,
                                       donor_minus_genuine=float(d.mean()), p80=np.percentile(boot, (10, 90)).tolist(),
                                       between_person_sd_peak=float(np.std([r["peak_A"] for r in rows])))
            print(split, a, {k: v for k, v in res[f"{split}_{a}"].items() if k != "rows"}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
