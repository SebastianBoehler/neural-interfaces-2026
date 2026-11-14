"""DESIGNDEC self-contained replay (ledger s239, test i): members as recorded (stored dev stack, A.mix), the stages in the
delivery order (shuffle.apply), rebalancing -> gate -> gated SessionEM -> gated TSEM, with the stage of stage.py installed.
Checks first that the installed stage with feedback off equals the stock stages (max |diff| 0). Then every variant in
VARIANTS (fb/boundary), scored by evalx (alone, + gated J filter, + gated true filter) against off.
Writes runs/designdec/replay_<base>_<order>.npz (recording layout).
Usage: [BASE=B2_pool] [VARIANTS=rep/J,ext/J,fwd/J,rep/true,ext/true] python studies/14nov/designdec/replay.py within0 [...]"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/../../13nov")
import stage as ST  # noqa: E402
import evalx as EX  # noqa: E402
import shuffle as S  # noqa: E402
A, gm, p0 = S.A, S.gm, S.p0
BASE = os.environ.get("BASE", "B2_pool"); VARIANTS = os.environ.get("VARIANTS", "rep/J,ext/J,fwd/J,rep/true,ext/true").split(",")
OUT = ST.C.REPO + "/runs/designdec"; os.makedirs(OUT, exist_ok=True)
stock_em, stock_ts = gm.em_g, gm.tsem_g; ST.install(gm)


def chain():
    b = gm.M.rebal(A.mix(BASE, p0, A.FAMS)[0], p0, None); g = gm.gate(b, p0); return gm.tsem_g(gm.em_g(b, p0, g), p0, g), b, g


def main():
    print(f"DESIGNDEC replay (s239 i): base {BASE}, members as recorded, stages in the delivery order; variants {VARIANTS}", flush=True)
    for order in sys.argv[1:]:
        sc, sd = order.rstrip("0123456789"), int(order[len(order.rstrip("0123456789")):]); P = S.perm_for(sc, sd)[p0]
        assert (P == ST.C.PERMS[f"{sc}|{sd}|{p0}"]).all(), "shuffle.perm_for and runs/shuffle_perms.npz disagree"
        S.apply({p: (P if p == p0 else np.arange(S.N[p])) for p in S.PR}); inv = np.argsort(P); outs = {}
        ST.CFG.update(fb="off", bnd="J"); t0 = time.time(); off, b, g = chain()
        ref = stock_ts(stock_em(b, p0, g), p0, g); d = np.abs(off - ref).max(); assert d == 0, f"installed stage, feedback off, differs from stock: {d}"
        print(f"{order}: no-op check OK (max |diff| {d}); off {time.time() - t0:.0f}s", flush=True); outs["off"] = off[inv]
        for v in VARIANTS:
            fb, bnd = v.split("/"); ST.CFG.update(fb=fb, bnd=bnd); ST.STATE["latch"] = []; t0 = time.time()
            outs[v] = chain()[0][inv]; lat = ST.STATE["latch"]
            print(f"{order}: {v} {time.time() - t0:.0f}s, latched {sum(w is not None for w in lat)}/{len(lat)}", flush=True)
        S.apply({p: np.arange(S.N[p]) for p in S.PR})
        np.savez(f"{OUT}/replay_{BASE}_{order}.npz", **{k: v.astype(np.float32) for k, v in outs.items()})
        EX.report(EX.person_scores(order, outs), order)


if __name__ == "__main__":
    main()
