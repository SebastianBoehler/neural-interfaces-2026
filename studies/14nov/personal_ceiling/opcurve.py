"""THE CHAIN'S OPERATING CURVE (ledger s252 amendment 2b): per order and session, at delivery position t (40, 80, 120, end)
the windows delivered before t are ranked and the top k handed over with their labels; purity = share of correct labels.
Rankings (labels in brackets):
  maxp   max posterior of the r15 chain, base mean (plabels_r15 '<order>|raw')             [chain argmax]
  agree  number of {eye net (runs/ocushuf/<order>, mean of 9 seeds), member mix (A.mix B2_pool, shipped families)}
         agreeing with the chain's argmax, ties by the chain's max posterior                [chain argmax]
  filt   max posterior of the gated count filter (plabels_r15 '<order>|filt')              [filter argmax]
The posteriors are those emitted when each window was delivered (causal). Output: purity at k = 10, 20, ... per scheme
(mean over the scheme's orders and the sessions with >= k windows before t), saved to runs/personal_ceiling/opcurve.npz.
Usage: python studies/14nov/personal_ceiling/opcurve.py"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "studies/14nov/rundetect")]
import numpy as np  # noqa: E402
import common as C  # noqa: E402
import splits as S  # noqa: E402

ORD = [f"within{i}" for i in range(12)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]
TS = (40, 80, 120, 10 ** 6); KS = np.arange(10, 200, 10); RANKS = ("maxp", "agree", "filt")


def sources(o, mix):
    PL = np.load(S.OUT / "plabels_r15.npz"); P, Pf = PL[f"{o}|raw"], PL[f"{o}|filt"]
    eye = np.mean([np.load(REPO / f"runs/ocushuf/{o}/oc_nomix-all_loso-s{k}.npz")["proba"] for k in range(9)], 0)
    a = P.argmax(1); ag = (eye.argmax(1) == a).astype(float) + (mix.argmax(1) == a)
    return {"maxp": (P.max(1), a), "agree": (ag + P.max(1) / 2, a), "filt": (Pf.max(1), Pf.argmax(1))}


def main():
    y = C.Y; mix = C.A.mix("B2_pool", "all_loso", C.A.FAMS)[0]; assert mix.shape == (C.N, 3)
    acc = {}  # (scheme, t, rank) -> {k: [purities]}
    for o in ORD:
        src = sources(o, mix); sc = o.rstrip("0123456789")
        for s in C.SESSIONS:
            rows = C.order_rows(o, s)
            for t in TS:
                pre = rows[:t]
                for r in RANKS:
                    score, lab = src[r]; top = pre[np.argsort(-score[pre], kind="stable")]; ok = lab[top] == y[top]
                    for k in KS[KS <= len(pre)]:
                        acc.setdefault((sc, t, r), {}).setdefault(int(k), []).append(ok[:k].mean())
    out = {}
    print("purity (%) of the top k windows handed over, mean over orders and sessions; t = delivery position ('end' = whole session)")
    for sc in ("within", "full", "runs"):
        for t in TS:
            print(f"{sc} t={'end' if t > 1000 else t}: " + " | ".join(f"{r} " + " ".join(f"{k}:{100 * np.mean(v):.1f}" for k, v in sorted(acc[sc, t, r].items()) if k in (10, 20, 30, 40, 60, 80, 100, 120, 140, 160)) for r in RANKS))
            for r in RANKS:
                d = acc[sc, t, r]; out[f"{sc}|{t}|{r}"] = np.array([[k, 100 * np.mean(v), len(v)] for k, v in sorted(d.items())])
    np.savez(S.OUT / "opcurve.npz", **out); print("saved", S.OUT / "opcurve.npz")


if __name__ == "__main__":
    main()
