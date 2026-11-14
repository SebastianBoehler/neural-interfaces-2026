"""v6.4 r15 reference check: the package's relabelled prefix at each refit (lean/src64 stages.reapply via
Ensemble6.relabel_prefix) against relabel.py's em_hook / ts_hook formulas (ledger s221 / s229), recomputed here from the
stage inputs the package stored: the SessionEM state is refitted from windows 0..t-1 (not read from the package), with
dev's experiments/session_em._softmax_lda and experiments/tsem._lda, the shipped constants relabel.py hard-codes
(asserted equal to the package's), and the boost profile of replay/mech.h at SHIFT 0. Recognition off (gates forced off).
Usage: python studies/14nov/r15_reference.py PKG [--order SCHEME:SEED:PERMS.npz] [--only K] [--raise-at N]
  --raise-at N: the relabelling raises at the refit at window N; asserts errors 1 and v5.1b's output from N on
Prints one JSON line per session (max |package - reference| over every refit, label agreement, relabel ms per refit) and a
RESULT line; exit 1 on a difference above 1e-10 or a refit count that differs from the schedule."""
import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import load_model  # noqa: E402
from session_em import _softmax_lda  # noqa: E402  (relabel.py's SE._softmax_lda)
from tsem import _lda  # noqa: E402  (relabel.py's G.T._lda)

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
EM = dict(lam=0.5, n0=60, m0=20, shrink=0.7, iters=2)  # relabel.py em_hook
TS = dict(lam=0.15, n0=60, rho=0.7, shrink=0.7, m0=20, nmin=15)  # relabel.py ts_hook


def h(t, T):  # replay/mech.h, SHIFT 0
    return float(np.clip((2 * T - t) / T, 0, 1)) if t >= 0 else 0.0


def reference(F, b, g, X, P, t, z):
    """relabel.py's R for one session at the refit at t: F EM view, b EM inputs, g gates, X TSEM view, P TSEM inputs."""
    nmin, bem, bts, T = int(z["em_nmin"]), float(z["em_boost"]), float(z["tsem_boost"]), float(z["eb_T"])
    q = b[:t + 1].copy()
    if t >= nmin:
        A, B = F[:t], b[:t]; m, sd = A.mean(0), np.maximum(A.std(0), 1e-3); Zm = (A - m) / sd; Q = B
        for _ in range(EM["iters"]):
            mu = (Q / (Q.sum(0, keepdims=True) + EM["m0"])).T @ Zm; C = np.cov((Zm - Q @ mu).T)
            C = (1 - EM["shrink"]) * C + EM["shrink"] * np.eye(len(C)) * np.trace(C) / len(C); Ci = np.linalg.inv(C)
            Q = B * _softmax_lda(Zm, mu, Ci) ** EM["lam"]; Q = Q / Q.sum(1, keepdims=True)
        e = EM["lam"] * t / (t + EM["n0"]) * (1 + bem * (h(t, T) * g[t]))
        q = q * _softmax_lda((F[:t + 1] - m) / sd, mu, Ci) ** e; q = q / q.sum(1, keepdims=True)
    Rem = q[:t]; R = Rem.copy(); lab, conf = P.argmax(1), P.max(1)
    if t >= TS["nmin"]:
        A = X[:t]; m, sd = A.mean(0), np.maximum(A.std(0), 1e-3); Zm = (A - m) / sd; k_ = max(2, int(TS["rho"] * t / 3)); sel = []
        for c in range(3):
            ik = np.flatnonzero(lab[:t] == c); sel.append(ik[np.argsort(-conf[ik])[:k_]])
        if min(len(x) for x in sel) >= 2:
            mu = np.stack([Zm[x].sum(0) / (len(x) + TS["m0"]) for x in sel]); Rr = np.concatenate([Zm[x] - mu[c] for c, x in enumerate(sel)])
            Cv = np.cov(Rr.T); Cv = (1 - TS["shrink"]) * Cv + TS["shrink"] * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
            Lk = _lda((X[:t + 1] - m) / sd, mu, np.linalg.inv(Cv)); e = TS["lam"] * t / (t + TS["n0"]) * (1 + bts * (h(t, T) * g[t]))
            R = Rem * Lk[:t] ** e; R = R / R.sum(1, keepdims=True)
    return R


def main():
    pkg = Path(sys.argv[1]); d = protocol.load(); torch.set_num_threads(1); z = np.load(pkg / "weights.npz")
    assert int(z["v6_ft_relabel"]) == 1, "r15 is off in this package"
    for k, v in dict(em_lam=EM["lam"], em_n0=EM["n0"], em_m0=EM["m0"], em_shrink=EM["shrink"], em_iters=EM["iters"], tsem_lam=TS["lam"],
                     tsem_n0=TS["n0"], tsem_rho=TS["rho"], tsem_shrink=TS["shrink"], tsem_m0=TS["m0"], tsem_nmin=TS["nmin"]).items():
        assert float(z[k]) == float(v), f"package constant {k} = {z[k]} differs from relabel.py's {v}"
    order, ra = arg("--order", "recording"), arg("--raise-at", -1)
    ss = list(dict.fromkeys(d["stream"])); ss = [ss[arg("--only", 0)]] if "--only" in sys.argv else ss
    g_ = None
    if order != "recording":
        sc, sd, f = order.split(":"); g_ = np.load(f)[f"{sc}|{sd}|all_loso"]
    m = load_model(pkg, None, "r15"); ok, worst = True, 0.0
    for s in ss:
        rows = g_[d["stream"][g_] == s] if g_ is not None else np.flatnonzero(d["stream"] == s)
        X = np.nan_to_num(np.asarray(d["X"][rows], np.float32)); c = copy.deepcopy(m)
        for r in (c.route, getattr(c, "routeB", None)):
            if r is not None:
                r.contrast.tau = r.identity.tau = np.inf
        c.reset_state(); chain = c.tail6r if c.v6r else c.route.tail; got, ms, gates = {}, [], []
        orig = c.ft.refit

        def rec(t, relab_fn=None, orig=orig):
            if relab_fn is not None:
                if t == ra:
                    raise RuntimeError("forced test exception in the relabelling")
                t0 = time.perf_counter(); got[t] = relab_fn(t); ms.append(1e3 * (time.perf_counter() - t0))
            return orig(t, relab_fn)
        c.ft.refit = rec
        out, o5 = [], []
        for x in X:
            out.append(np.asarray(c.step(x))); o5.append(c.trace["out_v5"]); gates.append(chain.em.gate)
        out, o5 = np.array(out), np.array(o5); log = dict(c.log6)
        F, b = np.asarray(chain.em.A), np.asarray(chain.em.B); U = np.asarray(chain.tsem.U)
        P = np.zeros((len(chain.tsem.lab), 3)); P[np.arange(len(P)), chain.tsem.lab] = chain.tsem.conf  # argmax / max as stored
        diff, agree, changed = 0.0, [], 0
        emitted = np.asarray(c.ft.lab)  # the labels as emitted (what s15 would train on)
        for t, G in got.items():
            changed += int((G.argmax(1) != emitted[:t]).sum())
            R = reference(F, b, np.asarray(gates), U, P, t, z); dd = float(np.abs(G - R).max()) if np.isfinite(G).all() else float("inf")
            diff = max(diff, dd); agree.append(float(np.mean(G.argmax(1) == R.argmax(1))))
        sched = len(range(int(z["v6_ft_t0"]), len(rows), int(z["v6_ft_every"])))
        if ra >= 0:
            good = log["err"] == 1 and np.array_equal(out[ra:], o5[ra:]) and diff <= 1e-10
        else:
            good = diff <= 1e-10 and len(got) == sched == c.ft.relabels and log["err"] == 0
        ok &= bool(good); worst = max(worst, diff)
        print(json.dumps(dict(session=s.split("/")[-1], windows=len(rows), refits=len(got), relabels=c.ft.relabels, schedule=sched,
                              max_diff=diff, min_label_agreement=min(agree) if agree else None, labels_changed_vs_emitted=changed, err=log["err"],
                              relabel_ms_per_refit=round(float(np.mean(ms)), 2) if ms else None, ok=bool(good))), flush=True)
    print("RESULT", json.dumps(dict(package=str(pkg), order=order, raise_at=ra, max_diff=worst, ok=bool(ok))), flush=True)
    assert ok, "r15 reference check failed"


if __name__ == "__main__":
    main()
