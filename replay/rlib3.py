"""Routing v3 (route2, 7 Nov): v2 gate (robust shape score, tau 0.22) on the UNROUTED shadow tail + PS1 personal head
injected into the B2 slot of a separate routed tail. Batch simulation on the dev bases:
  shadow  = B2 + TSEM of the base (pl.tsem(final)); gate scores from it (rlib2 / route_stream2 ops, windows < t);
  r_t     = mapping of the top score (t >= 60), j_t = top person;
  routed  = pl.pipe(mix'), mix'_t = (mix_t - r a core1_t + 3 r a h_j(t)) / (1 + 2 r a), a = 1/15 (core1's mix share);
  output  = routed tail for sessions with any r > 0, else the shadow (identical by construction in the stream).
PS1 = LR(C 0.1) on the person's training sessions in the fold (core1 PC256, each session mean/std standardised),
test rows causally standardised (route.causal_std)."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
import rlib2 as R2
L, RS = R2.L, R2.RS
import route as R  # noqa: E402
import pl  # noqa: E402

A, G3, TAU, T0 = 1.0 / 15, 3.0, 0.22, 60


def stdz(X):
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)


def gate_model(proto, f, oof, npc=256):
    """v2 gate of a fold (= rlib2.fold_model without its heads): names, normalised projected galleries Gn, basis Q."""
    tr, te = L.fold(proto, f); itr = np.load(f"{L.FE}/{proto}-{f}.npz")["idx_tr"]; st = L.STREAM[itr]; per, bias = {}, {}
    for s, (X, y) in tr.items():
        per.setdefault(L.PER(s), []).append((X, y)); Po = oof[itr[st == s]]
        bias.setdefault(L.PER(s), []).append(R2.contrast(X, L.weights(Po, len(y), "q70", y), npc) - R2.contrast(X, R2.onehot(y), npc))
    names = list(per); gal = np.stack([np.mean([R2.contrast(X, R2.onehot(y), npc) for X, y in per[p]], 0) for p in names])
    pop = gal.mean(0); b = np.mean([np.mean(bias[p], 0) for p in names], 0); Q = R2.basis(pop, b)
    return te, names, RS.gal_norm(gal, Q), Q, per, gal


def ps1_heads(per, names, C=0.1):
    return {p: LogisticRegression(C=C, max_iter=3000).fit(np.concatenate([stdz(X[:, :256]) for X, _ in per[p]]),
                                                         np.concatenate([y for _, y in per[p]])) for p in names}


STREAM_P = dict(rb_alpha=1.15, rb_k=5.0, rb_w=30, em_lam=0.5, em_n0=60.0, em_m0=20.0, em_shrink=0.7, em_iters=2, em_nmin=10,
                tsem_lam=0.15, tsem_rho=0.7, tsem_shrink=0.7, tsem_m0=20.0, tsem_n0=60.0, tsem_nmin=15, route_tau=TAU, route_ramp=0.05,
                route_rho=0.7, route_clip=4.0, route_t_start=T0, route_slot_a=A, route_slot_g=G3, route_v3=1)


def stream_params(names, gal, Q, heads, **kw):
    """route_stream3 parameter dict for a fold (dev) or the package."""
    return dict(STREAM_P, route_gal=gal, route_Q=Q, route_names=np.array(names), route_ps1_W=np.stack([heads[n].coef_ for n in names]),
                route_ps1_c=np.stack([heads[n].intercept_ for n in names]), **kw)


def rmap(Sc, mode, tau=TAU, t0=T0):
    """-> r (n,), j (n,) from running scores; mode lin10 / lin05 (ramp 0.10 / 0.05) / hard."""
    n = len(Sc); r = np.zeros(n); j = np.full(n, -1)
    for t in range(t0, n):
        if not np.isfinite(Sc[t]).all():
            continue
        k = int(np.argmax(Sc[t])); s = Sc[t, k]
        r[t] = {"lin10": np.clip((s - tau) / 0.10, 0, 1), "lin05": np.clip((s - tau) / 0.05, 0, 1), "hard": float(s >= tau)}[mode]; j[t] = k
    return r, j


def slot_mix(mix, c1, H, r, j):
    """Rows of one session: mix' with the routed person's PS1 head (H (P, n, 3)) at ramp r."""
    out = mix.copy()
    for t in np.flatnonzero(r > 0):
        out[t] = (mix[t] - r[t] * A * c1[t] + G3 * r[t] * A * H[j[t]][t]) / (1 - r[t] * A + G3 * r[t] * A)
    return out


def prepare(bn, cache=None):
    """Per protocol / test session: rows, gate scores Sc (from the shadow), PS1 head probabilities H, names."""
    PCC = np.load(R2.S + "/post_cache.npz"); oof = PCC[f"{bn}|tsem|all_loso"]; out = {}
    for proto, folds in R.FOLDS.items():
        idx = pl.B[f"idx|{proto}"]; pos = {int(g): k for k, g in enumerate(idx)}; Tsh = pl.tsem(pl.get(bn, "final", proto), proto)
        for f in folds:
            te, names, Gn, Q, per, _ = gate_model(proto, f, oof); hk = f"{proto}|{f}"
            if cache is not None and hk not in cache:
                cache[hk] = ps1_heads(per, names)
            heads = cache[hk] if cache is not None else ps1_heads(per, names)
            for s, (X, y, gi) in te.items():
                r = np.array([pos[int(g)] for g in gi]); Sc = R2.scores_session(X, Tsh[r], Gn, Q)
                Zc = R.causal_std(X[:, :256]); H = np.stack([heads[p].predict_proba(Zc) for p in names])
                out[(proto, s)] = dict(rows=r, Sc=Sc, H=H, names=names, X=X)
    return out


def run(bn, prep, mode, head_of=None):
    """-> ({proto: output}, {proto: shadow}, log). head_of(names, j_routed, own) -> index of the head to inject
    (None: the routed person's own = real arm; for the null: a wrong person's head with the same r)."""
    outs, shadows, log = {}, {}, []
    for proto in R.FOLDS:
        mix, c1 = pl.get(bn, "mix", proto), pl.get(bn, "core1", proto); Tsh = pl.tsem(pl.get(bn, "final", proto), proto)
        mixr = mix.copy(); routed = []
        for (pr, s), d in prep.items():
            if pr != proto:
                continue
            r, j = rmap(d["Sc"], mode)
            if (r > 0).any():
                jj = j.copy() if head_of is None else np.where(j >= 0, head_of(d["names"], j, L.PER(s)), -1)
                mixr[d["rows"]] = slot_mix(mix[d["rows"]], c1[d["rows"]], d["H"], r, jj); routed.append(d["rows"])
                who = sorted(set(d["names"][k] for k in j[r > 0]))
                log.append(f"{'NEW' if proto == 'all_loso' else 'SEEN'} {L.PER(s)}s{L.SNO(s)}->{'/'.join(who)} from t{np.flatnonzero(r > 0)[0]} "
                           f"mean r {r[r > 0].mean():.2f} on {int((r > 0).sum())}/{len(r)}")
        out = Tsh.copy()
        if routed:
            P = pl.pipe(mixr, proto); rr = np.concatenate(routed); out[rr] = P[rr]
        outs[proto], shadows[proto] = out, Tsh
    return outs, shadows, log
