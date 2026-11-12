"""NETEMB (ledger s72): the shipped SessionBN nets' penultimate features as a view for the session stage.
Per leave-one-person-out fold the net is trained with the shipped recipe; the classifier's input is captured while
the held-out person's sessions are streamed, and reduced to 64 components fitted on four training sessions streamed
through the same net. Saves proba, index, emb (n, 64). Usage (Galvani): python netemb.py TASK (arch = TASK % 4, seed = (TASK % 100) // 4; TASK >= 100: seen protocol) | smoke"""
import sys, time
from pathlib import Path
WS = Path(__import__("os").environ.get("NI26_WS", "/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026"))  # Leonardo: NI26_WS
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
OUT = WS / "runs/diag/netemb/probs"
ARCHS = {"fb": ("fb_sbn_rot05", 2.0), "fbraw": ("fbcnet_raw_sbn", 20.0), "fbl": ("fblight_ea_rot05_sbn", 5.0), "sinc": ("sincshallow_ea_ptm7r_rot05_sbn", 20.0)}


def capture(m, d, mask):
    """-> (proba (n,3), classifier inputs (n,F)) for the rows of mask, in row order; the capture is checked against the output."""
    head = [x for x in m.net.modules() if isinstance(x, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))][-1]; got, outs = [], []

    def hook(mod, inp, out):  # must return None: a returned value would replace the layer's output
        got.append(inp[0].detach().flatten(1).double().cpu().numpy()); outs.append(torch.softmax(out.detach().flatten(1)[:, :3].double(), 1).cpu().numpy())

    h = head.register_forward_hook(hook)
    try:
        P = protocol.predict_streams(m, d, mask, proba=True)
    finally:
        h.remove()
    n = int(mask.sum()); S = sum(len(g) for g in got) / n; assert S == int(S) and len(got) % int(S) == 0, f"{sum(len(g) for g in got)} classifier rows for {n} windows"
    S = int(S); E, O = [], []  # the stream predictor averages S time-shifted passes per session: average the captures the same way
    for i in range(0, len(got), S):
        assert len({len(g) for g in got[i:i + S]}) == 1, "shifted passes of one session differ in size"
        E.append(np.mean(got[i:i + S], 0)); O.append(np.mean(outs[i:i + S], 0))
    E, O = np.concatenate(E).astype(np.float32), np.concatenate(O); assert len(E) == n
    diff = np.abs(O - P).max(); assert diff < 1e-3, f"captured layer does not reproduce the stream output (max diff {diff:.4f})"
    assert np.unique(np.round(E, 5), axis=0).shape[0] > 0.99 * n, "captured features repeat across windows"
    return P, E


def run(arch, d, seed, out, epochs=0, proto="all_loso"):
    cfg, k = ARCHS[arch]; f = out / f"ne_{arch}-{proto}-s{seed}.npz"
    if f.exists():
        return print("exists", f.name, flush=True)
    idx, Ps, Es, t0, rng = [], [], [], time.time(), np.random.default_rng(seed)
    for fold, tr, te in protocol.folds(d, proto):
        m = SB.make(cfg, d, seed); assert type(m) is SB.SessionNet
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(k); P, E = capture(m, d, te)
        ss = rng.choice(np.unique(d["stream"][tr]), 4, replace=False); _, B = capture(m, d, tr & np.isin(d["stream"], ss))
        mu, sd = B.mean(0), np.maximum(B.std(0), 1e-6); pca = PCA(min(64, B.shape[1]), random_state=0).fit((B - mu) / sd); C = pca.transform((E - mu) / sd)
        C = np.pad(C, ((0, 0), (0, 64 - C.shape[1]))); idx.append(np.flatnonzero(te)); Ps.append(P); Es.append(C.astype(np.float32))
        print(f"{arch} s{seed} {fold} classifier input {E.shape[1]} -> {pca.n_components_} components (explained {pca.explained_variance_ratio_.sum():.2f}) {time.time() - t0:.0f}s", flush=True)
        if epochs:
            break
    idx, P, E = np.concatenate(idx), np.concatenate(Ps), np.concatenate(Es); o = np.argsort(idx); idx, P, E = idx[o], P[o], E[o]
    assert np.isfinite(P).all() and np.isfinite(E).all()
    np.savez(f, proba=P, index=idx, emb=E); print(f"TOTAL {f.name} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx]):.4f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    d = protocol.load()
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/netemb/smoke"; out.mkdir(parents=True, exist_ok=True)
        for arch in ARCHS:
            run(arch, d, 0, out, epochs=1); print("SMOKE PASS", arch, flush=True)
    else:
        t = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); proto = "seen_session" if t >= 100 else "all_loso"; t %= 100
        run(list(ARCHS)[t % 4], d, t // 4, OUT, proto=proto)
