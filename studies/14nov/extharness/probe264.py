"""Ledger s264: core1-like member (core feature set, each session centred at its own mean, standardised, LR C 0.05),
leave-one-person-out inside each external corpus, plus the cross-corpus reference (all PROTEUS people -> Scherer).
Test sessions are centred by their full own mean (primary) and by the causal running mean of harness.py (secondary).
Usage: python probe264.py [SMOKE=1 env: one fold per corpus, no output file] -> evidence/extprobe_14nov/s264.txt"""
import os, sys, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "3")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import corpus as CO  # noqa: E402
from harness import cell_ba, K0  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

SMOKE = bool(os.environ.get("SMOKE")); C_LR = 0.05


def proteus():
    path = CO.OUT + "/feat_proteus_core.npz"; src = CO.REPO + "/data/proteus_windows.npz"
    key = f"{os.path.getsize(src)}|{CO.SHRINK}|{CO.OV17}"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if str(z["key"]) == key:
            return {k: z[k] for k in z.files}
    sys.path.insert(0, CO.REPO + "/experiments"); from features import slow_waveform
    w = np.load(src, allow_pickle=True); X = w["X"].astype(np.float32)
    V = CO.tangent(X, CO.OV17); Xc = X - X.mean(1, keepdims=True); V = np.concatenate([V, slow_waveform(Xc, cutoff=1.0)], 1)
    d = dict(V=V, y=w["y"].astype(int), person=w["subject"].astype(str), session=w["session"].astype(str), key=key)
    np.savez(path, **d); return d


def center(V, person, session, causal=False):
    out = V.astype(np.float64); key = np.char.add(person, np.char.add("|", session))
    for k in np.unique(key):
        r = np.flatnonzero(key == k)
        out[r] -= (np.cumsum(out[r], 0) / (K0 + 1 + np.arange(len(r)))[:, None]) if causal else out[r].mean(0)
    return out


def fit_score(Vtr, ytr, Vte, yte, ste):
    mu, sd = Vtr.mean(0), Vtr.std(0) + 1e-6
    clf = LogisticRegression(C=C_LR, max_iter=500, tol=1e-4).fit((Vtr - mu) / sd, ytr)
    pred = clf.predict(((Vte - mu) / sd))
    return cell_ba(yte, pred), np.mean([cell_ba(yte[ste == s], pred[ste == s]) for s in dict.fromkeys(ste)])


def loso(name, lines):
    d = CO.feats(name, "core"); p, s, y = d["person"], d["session"], d["y"]
    Vc = center(d["V"], p, s); Vk = center(d["V"], p, s, causal=True); people = list(dict.fromkeys(p)); res = []
    for per in people[:1] if SMOKE else people:
        te = p == per; tr = ~te
        a = fit_score(Vc[tr], y[tr], Vc[te], y[te], s[te])
        # secondary: same training, test centred causally (reuse the fit by refitting is wasteful; refit once more)
        b = fit_score(Vc[tr], y[tr], Vk[te], y[te], s[te])
        res.append((per, te.sum(), a[0], a[1], b[1])); print(name, res[-1], flush=True)
    return summarise(name, res, lines)


def summarise(title, res, lines):
    arr = np.array([r[3] for r in res]); rng = np.random.default_rng(0)
    bs = np.sort([rng.choice(arr, len(arr)).mean() for _ in range(10000)])
    lines.append(f"## {title}: {len(res)} people, classes balanced; BA = mean over the person's sessions of per-session balanced accuracy")
    lines.append("person  windows  BA_pooled  BA_session_mean  BA_causal_centring")
    for per, n, a, a2, b in res:
        lines.append(f"{per}  {n}  {a:.1f}  {a2:.1f}  {b:.1f}")
    cs = np.mean([r[4] for r in res])
    lines.append(f"MEAN BA {arr.mean():.1f}  80% interval of the mean over people (bootstrap 10-90%) [{bs[1000]:.1f}, {bs[9000]:.1f}]  "
                 f"people spread 10-90% [{np.percentile(arr, 10):.1f}, {np.percentile(arr, 90):.1f}]  min {arr.min():.1f} max {arr.max():.1f}  "
                 f"causal-centring mean {cs:.1f}  mean windows/person {np.mean([r[1] for r in res]):.0f}\n")
    return arr.mean()


def cross(lines):
    P = proteus(); d = CO.feats("scherer2015", "core"); assert (np.bincount(P["y"]) > 0).all()
    Vp = center(P["V"], P["person"], P["session"]); Vs = center(d["V"], d["person"], d["session"])
    Vk = center(d["V"], d["person"], d["session"], causal=True); res = []
    for per in dict.fromkeys(d["person"]):
        te = d["person"] == per; res.append((per, te.sum()) + fit_score(Vp, P["y"], Vs[te], d["y"][te], d["session"][te])
                                           + (fit_score(Vp, P["y"], Vk[te], d["y"][te], d["session"][te])[1],))
        print("cross", res[-1], flush=True)
        if SMOKE: break
    lines.append(f"(reference: trained on all {len(P['y'])} PROTEUS windows / {len(set(P['person']))} people, class ids 0-2 as exported)")
    return summarise("CROSS-CORPUS PROTEUS -> Scherer2015", res, lines)


if __name__ == "__main__":
    lines = ["# s264 external within-corpus vs cross-corpus probe; member: core features (17 bands + slow block), per-session centring, LR C 0.05"]
    m = {"scherer2015": loso("scherer2015", lines), "shin2017b": loso("shin2017b", lines), "shin2017a": loso("shin2017a", lines)}
    m["cross"] = cross(lines)
    lines.append("SUMMARY " + " ".join(f"{k} {v:.1f}" for k, v in m.items()))
    if not SMOKE:
        open(CO.REPO + "/evidence/extprobe_14nov/s264.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))
