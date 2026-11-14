"""CO-TRAINING LABELS FROM THE EYE NET (ledger s258). Per order: classic aE (personal_ceiling/classic.py, imported) trained on
the four training runs' windows labelled by the eye net (runs/ocushuf/<order>/oc_nomix-all_loso-s0..8, 'proba' seed mean,
recording layout; trained on other people, so never on the test person). Arms:
  E<q>: eye argmax as label, per eye-argmax class the top q % of that class's training windows by eye max posterior
  A<q>: E<q> restricted to windows where the eye argmax equals the r15 chain argmax (plabels_r15 '<order>|raw'; selection first)
Output runs/cotrain/ec_<order>.npz: 'W|<session>|<pair>|idx', '...|<arm>|aE' (n_test, 3), '...|<arm>|n', '...|<arm>|fb' (fallback flag; no aE then), '...|<arm>|purity'.
Usage: python studies/14nov/cotrain/eyelab.py [--smoke] ORDER [ORDER ...]"""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "studies/14nov/personal_ceiling"))
import numpy as np  # noqa: E402
import classic as CL  # noqa: E402
import splits as S  # noqa: E402

QS = (30, 50, 70); ARMS = [f"{a}{q}" for a in "EA" for q in QS]; OUT = REPO / "runs/cotrain"


def eye(order):
    z = [np.load(REPO / f"runs/ocushuf/{order}/oc_nomix-all_loso-s{k}.npz") for k in range(9)]
    assert all((x["index"] == np.arange(len(x["index"]))).all() for x in z)
    return np.mean([x["proba"] for x in z], 0)


def select(trn, pe, pc, q, agree):
    """-> rows of trn kept (sorted)"""
    ye, conf = pe.argmax(1), pe.max(1); keep = []
    for c in range(3):
        r = trn[ye[trn] == c]; r = r[np.argsort(-conf[r], kind="stable")]; keep.append(r[:int(round(q / 100 * len(r)))])
    keep = np.sort(np.concatenate(keep))
    return keep[ye[keep] == pc[keep]] if agree else keep


def run(m, F, order, smoke):
    f = OUT / f"ec_{order}.npz"
    if f.exists() and not smoke:
        return print("exists", f)
    pe = eye(order); pc = np.load(S.OUT / "plabels_r15.npz")[f"{order}|raw"].argmax(1); ye = pe.argmax(1); res = {}; t0 = time.time()
    for s in S.sessions(m)[:1 if smoke else None]:
        for name, trn, tst in S.pairs(m, s):
            res[f"W|{s}|{name}|idx"] = tst
            for arm in ARMS:
                sub = select(trn, pe, pc, int(arm[1:]), arm[0] == "A"); fb = np.bincount(ye[sub], minlength=3).min() < 2
                res[f"W|{s}|{name}|{arm}|fb"] = np.array([fb])  # s258 amendment: fallback to the chain alone, no personal output
                if not fb:
                    res[f"W|{s}|{name}|{arm}|aE"] = CL.fit_predict(F, ye, sub, tst)["aE"]
                res[f"W|{s}|{name}|{arm}|n"] = np.array([len(sub)]); res[f"W|{s}|{name}|{arm}|purity"] = np.array([np.mean(ye[sub] == m["y"][sub])])
    if smoke:
        print("smoke", order, {a: (int(np.mean([res[k] for k in res if k.endswith(f"|{a}|n")])), round(float(np.mean([res[k] for k in res if k.endswith(f"|{a}|purity")])), 3)) for a in ARMS}, f"{time.time() - t0:.0f}s")
        return
    np.savez(f, **res); print("saved", f, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    smoke = sys.argv[1] == "--smoke"; orders = sys.argv[2 if smoke else 1:]; OUT.mkdir(exist_ok=True); m = S.meta(); F = CL.features(m)
    for o in orders:
        run(m, F, o, smoke)
