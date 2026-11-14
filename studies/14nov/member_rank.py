"""MEMBER RANK (ledger s211, ML panel's contrarian claim): does a member's accuracy after label-free per-session balancing
predict its chain effect better than its raw accuracy? Data: the fine-tuning form sweeps, where each form changes the four
fine-tuned SessionBN members and its chain change against the in-run gC5 control is scored (shufcombo files, V6Fr, new
people, six bases).
Per (order, form): d_chain = V6Fr(form) - V6Fr(gC5); d_raw / d_bal = member accuracy(form) - accuracy(gC5), mean over the
four families of the seed-mean posterior. Accuracy = person-equal balanced accuracy; 'bal' first shifts each session's
log posteriors by one bias vector so the session's mean prediction is uniform (label-free, whole session, as the panel).
Reports correlations of d_chain with d_raw and with d_bal, pooled and with order means removed.
Usage: python studies/14nov/member_rank.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; p0 = "all_loso"; Z = A.L.Z; IDX = A.pl.B[f"idx|{p0}"].astype(int)
y, per, ses = Z[f"y|{p0}"].astype(int), Z[f"person|{p0}"], Z[f"sess|{p0}"]; FAM = ("fbraw", "fb", "sinc", "fbl")
P18, CH = REPO + "/runs/ttt/probs18", [REPO + "/runs/leo_npz_sweep", REPO + "/runs"]


def bacc(P):
    return np.mean([np.mean([np.mean(P[(per == q) & (y == c)].argmax(1) == c) for c in range(3)]) for q in np.unique(per)])


def balance(P):
    L = np.log(np.clip(P, 1e-9, 1)); out = np.empty_like(L)
    for s in np.unique(ses):
        r = ses == s; b = np.zeros(3)
        for _ in range(100):
            q = np.exp(L[r] + b); q /= q.sum(1, keepdims=True); b -= np.log(q.mean(0) * 3)
        out[r] = L[r] + b
    return np.exp(out - out.max(1, keepdims=True))


def member(o, tag, form):
    accs = []
    for f in FAM:
        fs = sorted(glob.glob(f"{P18}/ttt18{o}{tag}_{f}_{form}-{p0}-s*.npz"))
        if not fs:
            return None
        P = np.mean([np.load(x)["proba"] for x in fs], 0); assert (np.load(fs[0])["index"].astype(int) == IDX).all()
        accs.append((bacc(P), bacc(balance(P))))
    return np.mean(accs, 0)


def chain(o, tag, form):
    for d in CH:
        fs = glob.glob(f"{d}/shufcombo_{o}_ocu_{form}_all_ge_hb_ts_gp0_ttt18{o}{tag}.npz")
        if fs:
            z = np.load(fs[0]); b = sorted({k.split("|")[0] for k in z.files}); return np.mean([z[f"{x}|V6Fr"] for x in b])
    return None


rows = []
for tag in ("LS", "LS3", "LF", "KF"):
    for o in [f"within{i}" for i in range(12)]:
        forms = sorted({os.path.basename(f).split("_")[2].split("-")[0] for f in glob.glob(f"{P18}/ttt18{o}{tag}_fb_*-{p0}-s0.npz")} - {"base", "gC5"})
        c0, m0 = chain(o, tag, "gC5"), member(o, tag, "gC5")
        if c0 is None or m0 is None:
            continue
        for fm in forms:
            c, m = chain(o, tag, fm), member(o, tag, fm)
            if c is not None and m is not None:
                rows.append((tag, o, fm, c - c0, 100 * (m[0] - m0[0]), 100 * (m[1] - m0[1])))
print(f"MEMBER RANK: {len(rows)} (order, form) pairs from the sweeps; d_chain vs d_raw / d_bal (member accuracy points)")
R = np.array([r[3:] for r in rows]); grp = np.array([f"{r[0]}{r[1]}" for r in rows])
dm = R.copy()
for g in np.unique(grp):
    dm[grp == g] -= R[grp == g].mean(0)
cc = lambda a, b: np.corrcoef(a, b)[0, 1]
rk = lambda a: np.argsort(np.argsort(a))
print(f"  pooled: corr(d_chain, d_raw) {cc(R[:, 0], R[:, 1]):+.2f}, corr(d_chain, d_bal) {cc(R[:, 0], R[:, 2]):+.2f} | Spearman {cc(rk(R[:, 0]), rk(R[:, 1])):+.2f} / {cc(rk(R[:, 0]), rk(R[:, 2])):+.2f}")
print(f"  order means removed: corr(d_chain, d_raw) {cc(dm[:, 0], dm[:, 1]):+.2f}, corr(d_chain, d_bal) {cc(dm[:, 0], dm[:, 2]):+.2f}")
print(f"  mean d_raw {R[:, 1].mean():+.2f}, d_bal {R[:, 2].mean():+.2f}, d_chain {R[:, 0].mean():+.2f}; slope chain per bal point {np.polyfit(dm[:, 2], dm[:, 0], 1)[0]:+.3f}, per raw point {np.polyfit(dm[:, 1], dm[:, 0], 1)[0]:+.3f}")
for tag in ("LS", "LS3", "LF", "KF"):
    k = np.array([r[0] == tag for r in rows])
    if k.sum() > 3:
        print(f"  {tag} ({k.sum()} pairs): corr raw {cc(dm[k, 0], dm[k, 1]):+.2f}, bal {cc(dm[k, 0], dm[k, 2]):+.2f}")
