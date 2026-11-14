"""PERSONAL CEILING (b) (ledger s246): population member nets fully fine-tuned on the person's own labelled runs.
Per task: arch = ttt5.ORDER[task % 4] (fbraw, fbl, sinc, fb), seed = task // 4. For each all_loso fold the net is
trained on the other nine people exactly as ttt7 trains it (ttt5.make, m.fit, m.set_k), then:
  W  for every session of the held-out person and every split of splits.pairs: the session's rows are ordered
     [train runs, test runs] and EA-aligned on themselves (non-causal); a copy of the net (eval-mode SessionBN, so the
     statistics are the session's expanding ones) is fine-tuned on the train rows with TRUE labels (all parameters,
     AdamW 5e-4, full batch, ttt5.strong augmentation each step); predictions for the test rows at 60 and 200 steps,
     plus the frozen net ('base') on the same inputs.
  X  P04 / P10 only: the copy is fine-tuned on all of session 1 (same recipe) and predicts sessions 2 and 3 with the
     same [train runs, test runs] arrangement as W (only the label source differs), and each whole session in
     recording order ('all').
Output runs/personal_ceiling/ft_<arch>_s<seed>.npz: '<W|X>|<session>|<pair>|<variant>' -> (n, 3), '...|idx' -> rows.
Usage (Leonardo): python studies/14nov/personal_ceiling/ftnet.py TASK | smoke"""
import copy, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts"), str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import splits as S  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("ttt5", WS / "studies/12nov/ttt5.py"); T5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T5)
STEPS, LR = (60, 200, 400), 5e-4  # 400: secondary (added after the smoke, before any real result)


def finetune(net, X, y, seed, w=None, steps=STEPS):
    """-> {steps: fine-tuned copy}; X train inputs on the device, y labels (true, or pseudo in s252), w optional sample
    weights (s252 soft forms: weighted mean CE; None = plain CE as in s246)"""
    net2 = copy.deepcopy(net).eval(); opt = torch.optim.AdamW(net2.parameters(), lr=LR, weight_decay=0.0)
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); yt = torch.from_numpy(y).to(X.device); out = {}
    wt = None if w is None else torch.from_numpy(w).float().to(X.device)
    for k in range(1, max(steps) + 1):
        opt.zero_grad(); z = net2(T5.strong(X, gen))[:, :3]
        loss = F.cross_entropy(z, yt) if wt is None else (wt * F.cross_entropy(z, yt, reduction="none")).sum() / wt.sum()
        loss.backward(); opt.step()
        if k in steps:
            out[k] = copy.deepcopy(net2).eval()
    assert torch.isfinite(loss), "fine-tuning diverged"
    return out


def proba(net, X):
    with torch.no_grad():
        return torch.softmax(net(X)[:, :3].double(), 1).cpu().numpy()


def run(arch, seed, smoke=False):
    d = protocol.load(); m0 = S.meta(); assert (m0["y"] == d["y"][:len(m0["y"])]).all(); res = {}; t0 = time.time()
    for fold, tr, te in protocol.folds(d, "all_loso"):
        if smoke and fold != "P04":
            continue
        m = T5.make(arch, d, seed)
        if smoke:
            m.epochs = 1
        m.fit(np.flatnonzero(tr)); m.set_k(T5.ARCHS[arch][1]); net = m.net.eval()
        al = lambda r: torch.from_numpy(m._aligned(r, causal=False)).to(m.dev)  # noqa: E731
        for s in [s for s in S.sessions(m0) if s.startswith(fold)]:
            for name, trn, tst in S.pairs(m0, s):
                X = al(np.concatenate([trn, tst])); n = len(trn); nets = finetune(net, X[:n], d["y"][trn], seed)
                res[f"W|{s}|{name}|idx"] = tst; res[f"W|{s}|{name}|base"] = proba(net, X)[n:]
                for k, nk in nets.items():
                    res[f"W|{s}|{name}|ft{k}"] = proba(nk, X)[n:]
                if smoke:
                    break
            print(f"{arch} s{seed} {fold} W {s} {time.time() - t0:.0f}s", flush=True)
        for p, s_tr, s_te in S.cross(m0):
            if p != fold:
                continue
            r1 = np.flatnonzero(m0["sess"] == s_tr); nets = finetune(net, al(r1), d["y"][r1], seed)
            for name, trn, tst in S.pairs(m0, s_te) + [("all", np.array([], int), np.flatnonzero(m0["sess"] == s_te))]:
                X = al(np.concatenate([trn, tst])); n = len(trn); res[f"X|{s_te}|{name}|idx"] = tst; res[f"X|{s_te}|{name}|base"] = proba(net, X)[n:]
                for k, nk in nets.items():
                    res[f"X|{s_te}|{name}|ft{k}"] = proba(nk, X)[n:]
            print(f"{arch} s{seed} {fold} X {s_tr}->{s_te} {time.time() - t0:.0f}s", flush=True)
    for k, v in res.items():
        assert np.isfinite(v).all(), k
    acc = {v: np.mean([S.cell_score(m0, res[k.rsplit('|', 1)[0] + '|idx'], res[k]) for k in res if k.startswith("W|") and k.endswith("|" + v)]) for v in ("base", "ft60", "ft200", "ft400")}
    print("W mean cell BA", {k: round(v, 1) for k, v in acc.items()}, flush=True)
    out = S.OUT / ("smoke" if smoke else ""); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"ft_{arch}_s{seed}.npz", **res); print("saved", out / f"ft_{arch}_s{seed}.npz", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "smoke":
        run("fbl", 0, smoke=True); print("SMOKE OK", flush=True)
    else:
        t = int(sys.argv[1]); run(T5.ORDER[t % len(T5.ORDER)], t // len(T5.ORDER))
