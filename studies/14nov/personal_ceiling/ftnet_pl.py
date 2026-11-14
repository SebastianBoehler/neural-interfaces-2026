"""LABEL-FREE CEILING (b) (ledger s252): ftnet.py's fine-tuning (200 steps) with the r15 chain's pseudo-labels of the
four training runs instead of the true labels. Task t: family ttt5.ORDER[t % 4], seed (t // 4) % 2, order group t // 8
(GROUPS). Forms as classic_pl.py (hard / soft / filt, from runs/personal_ceiling/plabels_r15.npz). The population net,
the inputs and the arrangement [train runs, test runs] are exactly ftnet.py's W part.
Output runs/personal_ceiling/ftpl_<family>_s<seed>_g<group>.npz: 'W|<session>|<pair>|<order>|<form>' -> (n, 3), '...|idx'.
Usage (Leonardo): python studies/14nov/personal_ceiling/ftnet_pl.py TASK | smoke"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import ftnet as FN  # noqa: E402
import splits as S  # noqa: E402
from classic_pl import FORMS, labels  # noqa: E402

GROUPS = (("within0", "within1", "full0"), ("full1", "runs0", "runs1"))


def run(arch, seed, group, smoke=False):
    out = S.OUT / ("smoke" if smoke else ""); f = out / f"ftpl_{arch}_s{seed}_g{group}.npz"
    if f.exists():  # galv_array requeues preempted tasks: skip finished outputs
        return print("exists", f, flush=True)
    d = FN.protocol.load(); m0 = S.meta(); PL = np.load(S.OUT / "plabels_r15.npz"); res = {}; t0 = time.time()
    for fold, tr, te in FN.protocol.folds(d, "all_loso"):
        if smoke and fold != "P04":
            continue
        m = FN.T5.make(arch, d, seed)
        if smoke:
            m.epochs = 1
        m.fit(np.flatnonzero(tr)); m.set_k(FN.T5.ARCHS[arch][1]); net = m.net.eval()
        for s in [s for s in S.sessions(m0) if s.startswith(fold)]:
            for name, trn, tst in S.pairs(m0, s)[:1 if smoke else None]:
                X = torch.from_numpy(m._aligned(np.concatenate([trn, tst]), causal=False)).to(m.dev); n = len(trn); res[f"W|{s}|{name}|idx"] = tst
                for o in GROUPS[group]:
                    for form in FORMS:
                        yp, w = labels(PL, o, form)
                        nk = FN.finetune(net, X[:n], yp[trn], seed, None if w is None else w[trn], steps=(200,))[200]
                        res[f"W|{s}|{name}|{o}|{form}"] = FN.proba(nk, X)[n:]
            print(f"{arch} s{seed} g{group} {fold} {s} {time.time() - t0:.0f}s", flush=True)
    for k, v in res.items():
        assert np.isfinite(v).all(), k
    out.mkdir(parents=True, exist_ok=True); np.savez(f, **res); print("saved", f, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "smoke":
        run("fbl", 0, 0, smoke=True); print("SMOKE OK", flush=True)
    else:
        t = int(sys.argv[1]); run(FN.T5.ORDER[t % 4], (t // 4) % 2, t // 8)
