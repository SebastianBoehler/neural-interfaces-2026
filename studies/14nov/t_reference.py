"""v6.5 T reference check (personal TSEM template prior, ledger PV2 / PV4, route4 KAPPA path with pview's denominator fix).
Gates as shipped (T acts on routed windows only). Per session, on the package with T on:
  (a) every step of RouteB's routed-tail TSEM against pview chain.tsem_lik_prior's form, recomputed from the inputs the
      step received (view, input posterior, gate, routed ramp r and person j): class means (sum + kappa r pri[j]) /
      (n + m0 + kappa r), dev's experiments/tsem._lda, the stock exponent; unrouted windows use no prior;
  (b) the stored prior of the session's person against route4's expression, recomputed here from the person's public
      sessions (rlib3.stdz class means of the package's own core1 TSEM coordinates, mean over sessions);
  (c) the T-off package's output on the same session: equal on every window before the first routed one.
Usage: python studies/14nov/t_reference.py PKG_T PKG_OFF [--order SCHEME:SEED:PERMS.npz] [--only K]
Prints one JSON line per session and a RESULT line; exit 1 on a TSEM difference above 1e-10, a prior difference above
1e-10, or a difference before the first routed window."""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import load_model  # noqa: E402
from tsem import _lda  # noqa: E402

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731


def h(t, T):  # replay/mech.h, SHIFT 0
    return float(np.clip((2 * T - t) / T, 0, 1))


def tsem_ref(U, conf, lab, u, q, g, t, kp, pm, z):
    """pview tsem_lik_prior (route4's KAPPA path, kp also in the denominator) and the stock exponent -> output for window t."""
    rho, shrink, m0, nmin, lam, n0, bts, T = (float(z["tsem_rho"]), float(z["tsem_shrink"]), float(z["tsem_m0"]), int(z["tsem_nmin"]),
                                              float(z["tsem_lam"]), float(z["tsem_n0"]), float(z["tsem_boost"]), float(z["eb_T"]))
    if t < nmin:
        return q
    A_ = U[:t]; m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3); Zm, zz = (A_ - m) / sd, (u - m) / sd
    k_ = max(2, int(rho * t / 3)); sel = [np.flatnonzero(lab[:t] == k)[np.argsort(-conf[np.flatnonzero(lab[:t] == k)])[:k_]] for k in range(3)]
    if min(len(x) for x in sel) < 2:
        return q
    mu = np.stack([(Zm[x].sum(0) + kp * pm[k]) / (len(x) + m0 + kp) for k, x in enumerate(sel)])
    R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)]); Cv = np.cov(R.T)
    Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
    e = lam * t / (t + n0) * (1 + bts * (h(t, T) * g))
    o = q * _lda(zz[None], mu, np.linalg.inv(Cv))[0] ** e
    return o / o.sum()


def prior_ref(m, d, person, rows_of):
    """route4: mean over the person's public sessions of stack(stdz(X[:, :128])[y == k].mean(0)), X the TSEM coordinates."""
    V0, vm, out = np.asarray(m.view_V[0]), np.asarray(m.view_m), []
    for s, rows in rows_of(person):
        c = copy.deepcopy(m); c.reset_state(); U = []
        for x in np.nan_to_num(np.asarray(d["X"][rows], np.float32)):
            U.append((c.core1.step(x)[1] - vm) @ V0)
        U = np.asarray(U); U = (U - U.mean(0)) / np.maximum(U.std(0), 1e-6); y = d["y"][rows].astype(int)
        out.append(np.stack([U[y == k].mean(0) for k in range(3)]))
    return np.mean(out, 0)


def main():
    pkg, off = Path(sys.argv[1]), Path(sys.argv[2]); d = protocol.load(); torch.set_num_threads(1); z = np.load(pkg / "weights.npz")
    assert int(z["v6_tpri_on"]) == 1 and int(np.load(off / "weights.npz")["v6_tpri_on"]) == 0
    order = arg("--order", "recording"); ss = list(dict.fromkeys(d["stream"])); ss = [ss[arg("--only", 0)]] if "--only" in sys.argv else ss
    g_ = None
    if order != "recording":
        sc, sd, f = order.split(":"); g_ = np.load(f)[f"{sc}|{sd}|all_loso"]
    m, m0 = load_model(pkg, None, "T"), load_model(off, None, "Toff"); names = [str(n) for n in z["route_names"]]; ok, worst = True, 0.0
    rows_of = lambda q: [(s, np.flatnonzero(d["stream"] == s)) for s in dict.fromkeys(d["stream"]) if s.split("/")[-1].split("|")[0] == q]  # noqa: E731
    for s in ss:
        rows = g_[d["stream"][g_] == s] if g_ is not None else np.flatnonzero(d["stream"] == s)
        X = np.nan_to_num(np.asarray(d["X"][rows], np.float32)); c, c0 = copy.deepcopy(m), copy.deepcopy(m0); c.reset_state(); c0.reset_state()
        ts, rec = c.routeB.tail.tsem, []
        orig = ts.step

        def tap(u, q, gate, ts=ts, orig=orig):
            U, conf, lab, pr = np.asarray(ts.U), np.asarray(ts.conf), np.asarray(ts.lab), ts.prior
            out = orig(u, q, gate); rec.append((len(U), U, conf, lab, u, q, gate, pr, out)); return out
        ts.step = tap
        o, o0, routed = [], [], []
        for x in X:
            o.append(np.asarray(c.step(x))); o0.append(np.asarray(c0.step(x))); routed.append(bool((c.routeB.last.get("contrast") or (0, 0, 0))[2] > 0 or (c.routeB.last.get("identity") or (0, 0, 0))[2] > 0))
        o, o0 = np.array(o), np.array(o0); diff, n_pri = 0.0, 0
        for (t, U, conf, lab, u, q, gate, pr, out) in rec:
            kp, pm = (pr if pr is not None else (0.0, np.zeros((3, U.shape[1] if len(U) else len(u)))))
            n_pri += int(pr is not None and pr[0] > 0)
            diff = max(diff, float(np.abs(tsem_ref(U, conf, lab, u, q, gate, t, kp, pm, z) - out).max()))
        first = int(np.flatnonzero(routed)[0]) if any(routed) else len(rows)
        before = bool(np.array_equal(o[:first], o0[:first])); person = s.split("/")[-1].split("|")[0]
        pd = float(np.abs(prior_ref(m, d, person, rows_of) - z["v6_tpri"][names.index(person)]).max()) if person in names else None
        good = diff <= 1e-10 and before and (pd is None or pd <= 1e-10) and c.log6["err"] == 0 and n_pri > 0
        ok &= bool(good); worst = max(worst, diff)
        print(json.dumps(dict(session=s.split("/")[-1], windows=len(rows), first_routed=first, windows_with_prior=n_pri, tsem_max_diff=diff,
                              prior_max_diff=pd, equal_before_routing=before, labels_changed_vs_off=int((o.argmax(1) != o0.argmax(1)).sum()),
                              max_change_vs_off=float(np.abs(o - o0).max()), err=c.log6["err"], ok=bool(good))), flush=True)
    print("RESULT", json.dumps(dict(package=str(pkg), order=order, max_diff=worst, ok=bool(ok))), flush=True)
    assert ok, "T reference check failed"


if __name__ == "__main__":
    main()
