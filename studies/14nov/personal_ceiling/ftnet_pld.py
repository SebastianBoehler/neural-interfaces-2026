"""s265 variants of ftnet_pl.py (own outputs; never touches the 16 base files).
V3 (seeds 2,3 on s252's split, 16 tasks): task t: family ORDER[t % 4], seed 2 + (t // 4) % 2, group t // 8 -> runs/personal_ceiling_v3/.
V2 (delivery prefix, 8 tasks): task t: family ORDER[t % 4], seed t // 4. Per fold net as ftnet_pl; per session and order in D_ORDERS
the session's windows in delivery order (chain.npz pos), EA-aligned CAUSALLY (window j's input depends on windows <= j only), net
fine-tuned 200 steps on the first TCUT windows with the r15 filt pseudo-labels (soft weight = max posterior), predictions for the
windows from TCUT on. Output runs/personal_ceiling_v2/ftd_<family>_s<seed>.npz: 'D|<session>|<order>|idx' (recording rows), '|p'.
Usage: python ftnet_pld.py v3|v2 TASK | smoke"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import ftnet as FN  # noqa: E402
import splits as S  # noqa: E402
from classic_pl import FORMS, labels  # noqa: E402
from ftnet_pl import GROUPS  # noqa: E402

SUF = os.environ.get("S265_SUF", "")  # "_leo": the Leonardo copies write to runs/personal_ceiling_v{2,3}_leo
D_ORDERS = ("within0", "within1", "full0", "full1", "runs0", "runs1"); TCUT = 120


def run_v3(arch, seed, group, smoke=False):
    out = S.REPO / f"runs/personal_ceiling_v3{SUF}" / ("smoke" if smoke else ""); f = out / f"ftpl_{arch}_s{seed}_g{group}.npz"
    if f.exists():
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
            print(f"v3 {arch} s{seed} g{group} {fold} {s} {time.time() - t0:.0f}s", flush=True)
    for k, v in res.items():
        assert np.isfinite(v).all(), k
    out.mkdir(parents=True, exist_ok=True); np.savez(f, **res); print("saved", f, flush=True)


def run_v2(arch, seed, smoke=False):
    out = S.REPO / f"runs/personal_ceiling_v2{SUF}" / ("smoke" if smoke else ""); f = out / f"ftd_{arch}_s{seed}.npz"
    if f.exists():
        return print("exists", f, flush=True)
    d = FN.protocol.load(); m0 = S.meta(); PL = np.load(S.OUT / "plabels_r15.npz"); CH = np.load(S.REPO / "runs/causalpers/chain.npz"); res = {}; t0 = time.time()
    for fold, tr, te in FN.protocol.folds(d, "all_loso"):
        if smoke and fold != "P04":
            continue
        m = FN.T5.make(arch, d, seed)
        if smoke:
            m.epochs = 1
        m.fit(np.flatnonzero(tr)); m.set_k(FN.T5.ARCHS[arch][1]); net = m.net.eval()
        for s in [s for s in S.sessions(m0) if s.startswith(fold)]:
            rows = np.flatnonzero(m0["sess"] == s)
            for o in D_ORDERS[:1 if smoke else None]:
                pos = CH[f"{o}|pos"]; dl = rows[np.argsort(pos[rows])]; assert (np.sort(pos[dl]) == pos[dl]).all() and len(dl) > TCUT + 10
                trn, tst = dl[:TCUT], dl[TCUT:]; yp, w = labels(PL, o, "filt")
                assert pos[trn].max() < TCUT <= pos[tst].min()  # no label or window >= t enters training
                X = torch.from_numpy(m._aligned(dl, causal=True)).to(m.dev)
                nk = FN.finetune(net, X[:TCUT], yp[trn], seed, w[trn], steps=(200,))[200]
                res[f"D|{s}|{o}|idx"] = tst; res[f"D|{s}|{o}|p"] = FN.proba(nk, X)[TCUT:]
            print(f"v2 {arch} s{seed} {fold} {s} {time.time() - t0:.0f}s", flush=True)
    for k, v in res.items():
        assert np.isfinite(v).all(), k
    out.mkdir(parents=True, exist_ok=True); np.savez(f, **res); print("saved", f, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "smoke":
        run_v2("fbl", 0, smoke=True); run_v3("fbl", 2, 0, smoke=True); print("SMOKE OK", flush=True)
    else:
        v, t = sys.argv[1], int(sys.argv[2]); a = FN.T5.ORDER[t % 4]
        run_v3(a, 2 + (t // 4) % 2, t // 8) if v == "v3" else run_v2(a, t // 4)
