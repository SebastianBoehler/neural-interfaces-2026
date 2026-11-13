"""TTT6 (13 Nov, ledger s108): arms that may raise the gain of the s101 fine-tuning for new people, package form.
Same nets, folds, augmentation and pseudo-labels as studies/12nov/ttt5.py (imported); per variant:
  fixA   ttt5's fixA again (in-run reference: 20 steps every 10 windows, each refit from the frozen net)
  cum10  warm start: each refit continues from the previous refit's weights, 10 steps (half the cost)
  cons2  consistency (KL) weight 2
  soft   soft pseudo-labels: CE against the chain's probabilities instead of its argmax weighted by confidence
  step5  a refit every 5 windows (twice the refits)
  head   only the SessionBN affine and the output layer are trained (cheaper backward)
  view2  two augmented views per step (CE and KL averaged over both)
Saved to runs/diag/ttt/probs6 as ttt6_<arch>_<variant>-<proto>-s<seed>.npz (rows sorted by index), plus base.
Usage (Leonardo): python studies/13nov/ttt6.py TASK (arch = ORDER[TASK % 4], seed = TASK // 4) | smoke"""
import copy, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]  # not studies/12nov: its profile.py shadows the stdlib
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("ttt5", WS / "studies/12nov/ttt5.py"); T5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T5)

OUT = WS / "runs/diag/ttt/probs6"; T0 = T5.T0; TAG = "ttt6"; SMOKE_OUT = "smoke6"; DRAW = 0  # ttt8.py etc. override these; DRAW 0 keeps every earlier run bit-identical
# variant -> (steps per refit, refit every, lr, scope, consistency weight, label form, warm start, views)
VARS = {"fixA": (20, 10, 5e-4, "all", 1.0, "hard", False, 1), "cum10": (10, 10, 5e-4, "all", 1.0, "hard", True, 1),
        "cons2": (20, 10, 5e-4, "all", 2.0, "hard", False, 1), "soft": (20, 10, 5e-4, "all", 1.0, "soft", False, 1),
        "step5": (20, 5, 5e-4, "all", 1.0, "hard", False, 1), "head": (20, 10, 5e-4, "head", 1.0, "hard", False, 1),
        "view2": (20, 10, 5e-4, "all", 1.0, "hard", False, 2)}


def params_of(net, scope):
    if scope == "head2":  # review 13 Nov: the personal-head recipe, the last two parameters (output weight and bias) + SessionBN affine
        sbn = {id(p) for mod in net.modules() if isinstance(mod, SB.SessionBN) for p in mod.parameters()}; last2 = {id(p) for p in list(net.parameters())[-2:]}
        assert any(p.dim() == 1 for p in list(net.parameters())[-2:]), "the last two parameters do not include a bias"
        return [p for p in net.parameters() if id(p) in sbn or id(p) in last2]
    if scope == "head":  # s108's form: never trained the output bias (selector takes only the final weight matrix)
        sbn = {id(p) for mod in net.modules() if isinstance(mod, SB.SessionBN) for p in mod.parameters()}
        last = [p for p in net.parameters() if p.dim() > 1 and p.shape[0] >= 3][-1]
        return [p for p in net.parameters() if id(p) in sbn or p is last]
    return list(net.parameters())


def stream(m, X, Pc, seed):
    """-> {variant: (n, 3)} for one session; X aligned causal inputs (n, C, T) on the device; Pc chain outputs (n, 3)."""
    net = m.net.eval(); n = len(X)
    with torch.no_grad():
        base = torch.softmax(net(X)[:, :3].double(), 1).cpu().numpy()
    out = {"base": base, **{v: base.copy() for v in VARS}}
    lab = torch.from_numpy(Pc.argmax(1)).to(X.device); wpl = torch.from_numpy(Pc.max(1)).float().to(X.device)
    soft = torch.from_numpy(Pc).float().to(X.device)
    for v, (steps, every, lr, scope, lam, form, warm, views) in VARS.items():
        ds = seed + 1000 * DRAW; torch.manual_seed(ds); gen = torch.Generator().manual_seed(ds); net2 = None  # DRAW > 0: fresh augmentation draws on the same nets
        for t in range(T0, n, every):
            if net2 is None or not warm:
                net2 = copy.deepcopy(net).eval()
            opt = torch.optim.AdamW(params_of(net2, scope), lr=lr, weight_decay=0.0)
            for _ in range(steps):
                opt.zero_grad(); z = net2(X[:t])[:, :3]; loss = 0.0
                for _ in range(views):
                    zs = net2(T5.strong(X[:t], gen))[:, :3]
                    if form == "hent":  # s154: hard labels on the confident half of the prefix, entropy minimisation on the rest
                        m = wpl[:t] >= wpl[:t].median(); pr = F.softmax(zs[~m], 1)
                        ce = ((wpl[:t][m] * F.cross_entropy(zs[m], lab[:t][m], reduction="none")).sum() / wpl[:t][m].sum().clamp_min(1e-6)
                              + (-(pr * torch.log(pr + 1e-8)).sum(1)).mean())
                    else:
                        ce = (-(soft[:t] * F.log_softmax(zs, 1)).sum(1).mean() if form == "soft" else
                              (wpl[:t] * F.cross_entropy(zs, lab[:t], reduction="none")).sum() / wpl[:t].sum().clamp_min(1e-6))
                    loss = loss + (ce + lam * F.kl_div(F.log_softmax(zs, 1), F.softmax(z.detach(), 1), reduction="batchmean")) / views
                loss.backward(); opt.step()
            hi = min(n, t + every)
            with torch.no_grad():
                out[v][t:hi] = torch.softmax(net2(X[:hi])[:, :3].double(), 1).cpu().numpy()[t:hi]
    return out


def run(arch, d, proto, seed, out, epochs=0):
    tag = f"{TAG}_{arch}"; names = ["base"] + list(VARS)
    if all((out / f"{tag}_{v}-{proto}-s{seed}.npz").exists() for v in names):
        return print("exists", tag, proto, seed, flush=True)
    got = {v: [] for v in names}; rows_all = []; t0 = time.time(); Pc_all = T5.chain_labels(proto)
    for fold, tr, te in protocol.folds(d, proto):
        m = T5.make(arch, d, seed)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(T5.ARCHS[arch][1]); rows = np.flatnonzero(te); st = d["stream"][rows]
        ref = protocol.predict_streams(m, d, te, proba=True)
        for s in dict.fromkeys(st):
            r = rows[st == s]; X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev)
            res = stream(m, X, np.stack([Pc_all[int(g)] for g in r]), seed)
            for v in names:
                got[v].append(res[v])
            rows_all.append(r)
        base = np.concatenate(got["base"][-len(dict.fromkeys(st)):])
        print(f"{tag} s{seed} {proto} {fold} base-vs-harness max diff {np.abs(base - ref).max():.2e} {time.time() - t0:.0f}s", flush=True)
        assert np.abs(base - ref).max() < 1e-6, "the streamed frozen net does not reproduce the harness"
        if epochs:
            break
    idx = np.concatenate(rows_all); o = np.argsort(idx)
    for v in names:
        P = np.concatenate(got[v])[o]; assert np.isfinite(P).all()
        np.savez(out / f"{tag}_{v}-{proto}-s{seed}.npz", proba=P, index=idx[o], labels=T5.LABEL_TAG)
        print(f"TOTAL {tag}_{v} s{seed} {proto} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx[o]]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load()
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/ttt" / SMOKE_OUT; out.mkdir(parents=True, exist_ok=True)
        run("fbl", d, "seen_session", 0, out, epochs=1); return print("SMOKE OK", flush=True)
    task = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        run(T5.ORDER[task % len(T5.ORDER)], d, proto, task // len(T5.ORDER), OUT)


if __name__ == "__main__":
    main()
