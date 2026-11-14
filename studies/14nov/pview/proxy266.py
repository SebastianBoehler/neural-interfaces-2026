"""s266: the s262 routed personal path with the package's label-free gate instead of the oracle (part a), and the shippable-constant
grid for the nested tuning (part b). Built on proxy_routed.routed (s262's code path; zero-change asserted below) and gate266 (the
gate's raw scores per session and order, runs/pview/gate266.npz, built by gate266.py).
MODE=a: arms (all ER inputs, shipped constants beta 0.10, G3 3, kappa 20 for the T arms):
  tail0               no ER, no personal parts (asserted == tproxy.npz base)
  ER_ora0 / ERT_ora0  gate stub: correct person, r = 1 from the first online window (asserted == routed262 ER / ERT, atol 1e-9)
  ER_ora10 / ERT_ora10  stub, r = 1 from the gate's first routable window (t0 = 10)
  ER_gate / ERT_gate  the real gate's r (tau 0.26, ramp 0.05, t0 10; r = 0 where the top person is not the session's own)
MODE=grid: ERT path, full factorial over G3 {2,3,4,6} x kappa {10,20,40} x tau {.22,.26,.30} x ramp {.03,.05,.08} x beta
  {.05,.10,.20,.30}, the real gate's r; configs with identical r arrays and (G3, kappa) share one pipeline run (beta is applied after it).
Usage: [BASES=..] [ORDERS=..] [MODE=a|grid] [TAG=..] python studies/14nov/pview/proxy266.py -> runs/pview/s266_<MODE>_<TAG><orders>.npz"""
import os, sys, time, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import proxy_routed as PR  # noqa: E402
import gate266 as GT  # noqa: E402
S, A, D, Z, p0, ONL = PR.S, PR.A, PR.D, PR.Z, PR.p0, PR.ONL
MODE = os.environ.get("MODE", "a"); TAG = os.environ.get("TAG", "")
GZ = np.load(D.OUT + "/gate266" + os.environ.get("GATE", "") + ".npz")
GRID = dict(g3=(2.0, 3.0, 4.0, 6.0), kappa=(10.0, 20.0, 40.0), tau=(0.22, 0.26, 0.30), ramp=(0.03, 0.05, 0.08), beta=(0.05, 0.10, 0.20, 0.30))
SHIPPED = dict(g3=3.0, kappa=20.0, tau=0.26, ramp=0.05, beta=0.10)
assert (SHIPPED["tau"], SHIPPED["ramp"]) == (GT.TAU, GT.RAMP) and SHIPPED["g3"] == PR.G3 and SHIPPED["beta"] == PR.BETA and SHIPPED["kappa"] == PR.KAP


def r_sessions(order, idx, kind, tau=GT.TAU, ramp=GT.RAMP):
    """{session: r per row of the session (store order)}: acquisition rows 0; online rows by `kind` (ora0, ora10, gate)."""
    sess = Z[f"sess|{p0}"]; out = {}
    for s in dict.fromkeys(sess):
        rows = np.flatnonzero(sess == s); gi = idx[rows]; on = ONL[gi]; n = int(on.sum()); r = np.zeros(len(rows))
        if kind == "ora0":
            r[on] = 1.0
        elif kind == "ora10":
            r[np.flatnonzero(on)[GT.T0:]] = 1.0
        else:
            j, sc, names = GZ[f"own|{order}|{s}|j"], GZ[f"own|{order}|{s}|s"], GZ[f"own|{order}|{s}|names"]; assert len(j) == n
            ok = (j == len(names) - 1) & np.isfinite(sc)  # the session's own person is the last gallery entry
            r[on] = np.where(ok, np.clip((np.nan_to_num(sc) - tau) / ramp, 0, 1), 0.0)
            if (tau, ramp) == (GT.TAU, GT.RAMP):
                assert np.allclose(r[on][ok], GZ[f"own|{order}|{s}|r"][ok], atol=1e-12), f"{order} {s}: the stored raw scores do not give Bio.step's r"
        out[s] = r
    return out


def cells():
    z262 = {}
    for order in PR.ORDERS:
        sc, sd = order.rstrip("0123456789"), order[len(order.rstrip("0123456789")):]
        P = S.perm_for(sc, int(sd)); S.apply(P); idx = Z[f"idx|{p0}"].astype(int)
        assert (idx == PR.PF[f"{sc}|{sd}|{p0}"]).all() and (idx == np.arange(D.N)[P[p0]]).all(), "the order is not the pinned order file's"
        eye = np.log(np.clip(np.mean([PR.ld(f"{D.REPO}/runs/ocushuf/{order}/oc_nomix-{p0}-s{k}.npz") for k in range(3)], 0), 1e-3, 1))[P[p0]]
        rest = np.mean([PR.ld(f"{D.REPO}/runs/restshuf/{order}/rest4_sh_r1-{p0}-s{k}.npz") for k in range(3)], 0)[P[p0]]
        ge0 = Z[f"GE|{p0}"]
        for bn in PR.BASES:
            mix0, a0 = A.mix(bn, p0, A.FAMS); c10 = A.members(bn, p0)["c1"]; tail0 = PR.ba_online(PR.routed(mix0, a0, c10, idx, False), idx)  # before the ER inputs are set
            try:
                A.EXTRA["rt_rest"] = {p0: rest}; Z[f"GE|{p0}"] = np.c_[ge0, eye]; mix, a = A.mix(bn, p0, A.FAMS, [("rt_rest", 1.0)]); assert abs(a - 1 / 16) < 1e-12
                c1 = A.members(bn, p0)["c1"]
                yield order, bn, idx, tail0, (mix, a, c1)
            finally:
                Z[f"GE|{p0}"] = ge0; A.EXTRA.pop("rt_rest", None); PR.R3.A = 1 / 15
    S.apply({q: np.arange(S.N[q]) for q in S.PR})


def mode_a():
    R, t0 = {}, time.time(); ref262 = {}
    for f in sorted(os.listdir(D.OUT)):  # s262's per-order files (the zero-change reference)
        if f.startswith("routed262_") and "smoke" not in f:
            z = np.load(f"{D.OUT}/{f}"); ref262.update({k: z[k] for k in z.files})
    for order, bn, idx, tail0, (mix, a, c1) in cells():
        R[order, bn, "tail0"] = tail0
        assert np.allclose(R[order, bn, "tail0"], PR.TP[f"{order}|{bn}|base"], atol=1e-9), f"{order} {bn}: tail0 differs from tproxy base"
        for arm, kind in (("ora0", "ora0"), ("ora10", "ora10"), ("gate", "gate")):
            rs = r_sessions(order, idx, kind)
            for name, kap in (("ER", 0.0), ("ERT", PR.KAP)):
                R[order, bn, f"{name}_{arm}"] = PR.ba_online(PR.routed(mix, a, c1, idx, True, kap, rs), idx)
                if arm == "ora0":  # zero-change: the stub through the new code path reproduces s262
                    k = f"{order}|{bn}|{name}"
                    assert k in ref262 and np.allclose(R[order, bn, f"{name}_ora0"], ref262[k], atol=1e-9), f"{order} {bn} {name}: stub {R[order, bn, f'{name}_ora0']} != s262 {ref262.get(k)}"
        b = R[order, bn, "tail0"]
        print(f"{order} {bn}: tail {b.mean():.2f} | " + " ".join(f"{k} {(R[order, bn, k] - b).mean():+.2f}" for k in ("ER_ora0", "ER_ora10", "ER_gate", "ERT_ora0", "ERT_ora10", "ERT_gate"))
              + f" [{time.time() - t0:.0f}s]", flush=True)
    return R


def mode_grid():
    R, t0 = {}, time.time(); cfgs = list(itertools.product(GRID["g3"], GRID["kappa"], GRID["tau"], GRID["ramp"]))
    for order, bn, idx, _, (mix, a, c1) in cells():
        done, nrun = {}, 0
        for g3, kap, tau, ramp in cfgs:
            rs = r_sessions(order, idx, "gate", tau, ramp); sig = (g3, kap, hash(b"".join(rs[s].tobytes() for s in rs)))
            if sig not in done:
                done[sig] = [PR.ba_online(o, idx) for o in PR.routed(mix, a, c1, idx, True, kap, rs, g3, GRID["beta"])]; nrun += 1
            for b_, v in zip(GRID["beta"], done[sig]):
                R[order, bn, f"g{g3:g}_k{kap:g}_t{tau:g}_r{ramp:g}_b{b_:g}"] = v
        print(f"{order} {bn}: {len(cfgs)} gate/head/prior configs, {nrun} pipeline runs [{time.time() - t0:.0f}s]", flush=True)
    return R


if __name__ == "__main__":
    R = mode_a() if MODE == "a" else mode_grid()
    os.makedirs(D.OUT, exist_ok=True)
    np.savez(f"{D.OUT}/s266_{MODE}_{TAG}{'-'.join(PR.ORDERS)}{'' if len(PR.BASES) == 6 else '_smoke'}.npz", **{f"{o}|{b}|{k}": v for (o, b, k), v in R.items()})
    print("written", flush=True)
