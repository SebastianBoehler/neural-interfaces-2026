"""Lane R4b threshold fit (phase 1 outputs, local). usage: fit_gate.py OUT_DIR B10_OUT_DIR SET DEST.json
Per (person, stream, order): proxy summary from the stock-path run `Gconf_off` (mean over post-recognition windows 10..250 of conf / agree / ent), gain = BA2(Gconf_on = R4) - BA2(hardcbB).
Spearman(proxy, gain), people-nested bootstrap 80 %; leave-one-person-out theta (stream gated on iff proxy < theta; value = gain if on else 0) and the full-sample theta; the open share.
hardcbB reference: net from the B10 run (B10_OUT_DIR, same seeds/orders), ds7 from OUT_DIR."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402

out, b10, setn, dest = sys.argv[1:5]
rng = np.random.default_rng(77)
off, on = S.load(out, setn, "Gconf_off"), S.load(out, setn, "Gconf_on")
ref = S.load(out, setn, "hardcbB") if setn == "ds7" else {k: S.streams(Path(b10) / f"hardcbB_k{k}" / "ret.npz", True) for k in range(19)}
COL = {"conf": 2, "agree": 4, "ent": 6}
rows = []  # person, proxy dict, gain, rec
for k in off:
    for s, e in off[k].items():
        if k in on and s in on[k] and k in ref and s in ref[k] and "pxy" in e:
            px = e["pxy"]
            post = px[:, 0] > 0
            if not post.any():
                rows.append((k, None, on[k][s]["ba2"] - ref[k][s]["ba2"]))
                continue
            r0 = int(np.flatnonzero(post)[0]); seg = px[r0 + 10:250]
            rows.append((k, {n: float(np.nanmean(seg[:, c])) if len(seg) else np.nan for n, c in COL.items()}, on[k][s]["ba2"] - ref[k][s]["ba2"]))
res = {"n_streams": len(rows), "unrecognised": sum(r[1] is None for r in rows), "gain_R4_mean": float(np.mean([r[2] for r in rows]))}
rec = [r for r in rows if r[1] is not None]
people = sorted({r[0] for r in rec})
for n in COL:
    x = np.array([r[1][n] for r in rec]); g = np.array([r[2] for r in rec]); who = np.array([r[0] for r in rec])
    ok = ~np.isnan(x)
    rho = spearmanr(x[ok], g[ok])[0]
    bs = []
    for _ in range(2000):
        pick = rng.choice(people, len(people)); idx = np.concatenate([np.flatnonzero((who == p) & ok) for p in pick])
        bs.append(spearmanr(x[idx], g[idx])[0])
    grid = np.quantile(x[ok], np.linspace(0.05, 0.95, 37))
    val = lambda th, m: np.where(x[m] < th, g[m], 0).mean()
    best = grid[int(np.argmax([val(t, ok) for t in grid]))]
    lo, loo = [], []
    for p in people:  # leave-one-person-out theta
        tr = ok & (who != p); te = ok & (who == p)
        t = grid[int(np.argmax([val(t_, tr) for t_ in grid]))]; loo.append((t, np.where(x[te] < t, g[te], 0).sum(), te.sum()))
    res[n] = dict(spearman=float(rho), spearman_i80=[float(np.nanpercentile(bs, 10)), float(np.nanpercentile(bs, 90))], theta_full=float(best),
                  gated_gain_in_sample=float(val(best, ok)), gated_gain_loo=float(sum(l[1] for l in loo) / sum(l[2] for l in loo)),
                  open_share=float((x[ok] < best).mean()), proxy_quantiles=[float(v) for v in np.quantile(x[ok], [.1, .25, .5, .75, .9])], loo_theta_range=[float(min(l[0] for l in loo)), float(max(l[0] for l in loo))])
Path(dest).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
