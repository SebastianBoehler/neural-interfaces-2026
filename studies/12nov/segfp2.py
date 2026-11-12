"""SEGFP2 (ledger s62b): stage-level fingerprints of stream cuts and run orders. Usage: python studies/12nov/segfp2.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay"]
import env  # noqa: E402
import session_em as SE  # noqa: E402
A = env.A; SC = A.SC; L = A.L; Z = L.Z; G = L.G
W = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); TASK = W["task"].astype(str)


def run_stages(mix, sess, GE, PC):
    b = L.rebal(mix, sess); e = SE.apply(GE, b, sess, lam=0.5, n0=60, m0=20, shrink=0.7, nmin=10); Lk, tt = G.tsem_lik(e, sess, PC, rho=0.7)
    return e, G.apply_lam(e, Lk, tt, 0.15, n0=60)


def variant(bn, p, kind):
    idx = Z[f"idx|{p}"].astype(int); s0 = Z[f"sess|{p}"]; itf, task = L.IFACE[idx], TASK[idx]; n = len(idx); mix = L.mixw(Z[f"{bn}|Ps|{p}"], Z[f"{bn}|ws|{p}"])
    order = np.arange(n); sess = s0.copy()
    if kind == "reset itf":
        sess = np.unique(np.char.add(np.char.add(s0.astype(str), "|"), itf), return_inverse=True)[1]
    elif kind == "reset task":
        sess = np.unique(np.char.add(np.char.add(s0.astype(str), "|"), task), return_inverse=True)[1]
    elif kind.startswith("order"):
        key = {"order Graz,BH": (itf != "Graz").astype(int), "order BH,Graz": (itf == "Graz").astype(int), "order task a-z": np.unique(task, return_inverse=True)[1]}[kind]
        order = np.lexsort((np.arange(n), key, s0))  # within a session: by block key, then acquisition order
    e, t = run_stages(mix[order], sess[order], Z[f"GE|{p}"][order], Z[f"pc128|{p}"][order]); inv = np.argsort(order)
    return e[inv], t[inv]


KINDS = ("session", "reset itf", "reset task", "order Graz,BH", "order BH,Graz", "order task a-z")
for bn in env.EXPL:
    R = {k: {p: variant(bn, p, k) for p in A.PROTOS} for k in KINDS}
    Jl = lambda k, l: 0.5 * SC.person(R[k]["all_loso"][l], "all_loso").mean() + 0.5 * SC.person(R[k]["seen_session"][l], "seen_session").mean()  # noqa: E731
    chk = L.chain(bn, "all_loso", stages=True); assert np.abs(chk[3] - R["session"]["all_loso"][1]).max() < 1e-12, "the re-implemented stages differ from the replay"
    print(f"{bn}: per session, acquisition order: SessionEM level J {Jl('session', 0):.2f}, TSEM level J {Jl('session', 1):.2f}   (sealed change after the fix: -0.72, -1.17, step -0.45)", flush=True)
    for k in KINDS[1:]:
        a, b = Jl(k, 0) - Jl("session", 0), Jl(k, 1) - Jl("session", 1); ok = abs(a + 0.72) <= 0.4 and abs(b + 1.17) <= 0.4 and abs((b - a) + 0.45) <= 0.3
        dn = SC.person(R[k]["all_loso"][1], "all_loso") - SC.person(R["session"]["all_loso"][1], "all_loso"); ds = SC.person(R[k]["seen_session"][1], "seen_session") - SC.person(R["session"]["seen_session"][1], "seen_session")
        print(f"   {k:15s}: SessionEM {a:+.2f}, TSEM {b:+.2f}, step {b - a:+.2f} {'MATCH' if ok else '     '} | TSEM level new {dn.mean():+.2f} seen {ds.mean():+.2f}", flush=True)
