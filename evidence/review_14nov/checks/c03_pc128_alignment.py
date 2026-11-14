"""shufcombo.py line 98-99 adds sign * (coreshuf order - coreshuf rec) to the stored pc128, assuming the rebuilt PCs equal
the stored ones up to sign. No assertion checks it. Per column: |corr| between stored pc128 and the rebuild, and how much
of the added delta lies outside what a per-column sign flip can map (orthogonal Procrustes fit on the recording rows)."""
import glob, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); Z = np.load(REPO + "/runs/replay_data/inputs.npz", allow_pickle=True)
for f in sorted(glob.glob(REPO + "/runs/coreshuf_p0*.npz")) + [REPO + "/runs/coreshuf_p4_within.npz"]:
    C = np.load(f)
    for p in ("all_loso", "seen_session"):
        ref, rec = Z[f"pc128|{p}"].astype(float), C[f"rec|{p}"].astype(float)
        c = np.array([np.corrcoef(ref[:, j], rec[:, j])[0, 1] for j in range(ref.shape[1])]); a = np.abs(c)
        rel = np.abs(np.sign(c) * rec - ref).mean(0) / np.abs(ref).mean(0)
        print(f"{os.path.basename(f)} {p}: |corr| min {a.min():.3f} p10 {np.percentile(a, 10):.3f} median {np.median(a):.3f}; columns with |corr| < 0.9: {(a < 0.9).sum()}, < 0.5: {(a < 0.5).sum()} of {len(a)}; "
              f"first column below 0.9: {int(np.argmax(a < 0.9)) if (a < 0.9).any() else None}; mean |sign*rec - stored| / |stored| per column: median {np.median(rel):.2f}, max {rel.max():.2f}")
        for k in [k for k in C.files if k.endswith("|" + p) and not k.startswith("rec")][:1]:
            d = (C[k].astype(float) - rec); print(f"    {k}: delta rms per column / stored sd per column: median {np.median(d.std(0) / ref.std(0)):.2f}; share of delta energy in columns with |corr| < 0.9: {(d[:, a < 0.9] ** 2).sum() / (d ** 2).sum():.2f}")
