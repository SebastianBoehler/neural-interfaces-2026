"""SSL1 (13 Nov, ledger s147): self-supervised trunks from the unlabelled PROTEUS recordings, then the usual training.
Per left-out person q (all_loso folds), only the nine training people's recordings are used, labels ignored:
their task windows and the rest period before each Graz cue (data/proteus_windows_start-4.5.npz, aligned with the
official windows' statistics). Objective: SimCLR (NT-Xent, temperature 0.1) between two augmented views of a window
(time shift up to 15 samples, noise 0.2 sd, 10 % channels dropped, amplitude x U(0.8, 1.25)), on the net's output
layer widened to 64 units; 1500 steps of 256 windows, AdamW 1e-3. The trunk (everything but the output layer) then
initialises the shipped FBLight SessionBN recipe (fblight_ea_rot05_sbn), trained and scored as usual; the twin is the
same recipe from a fresh trunk, same seed. Outputs runs/diag/ssl/probs/ssl_fbl_{ssl,twin}-all_loso-s<seed>.npz.
Usage (Leonardo): python studies/13nov/sslnet.py SEED | smoke"""
import os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
import deep  # noqa: E402
CFG, K = "fblight_ea_rot05_sbn", 5.0; OUT = WS / "runs/diag/ssl/probs"; PRE = WS / "runs/diag/ssl/pretrain"


def aug(X, g):
    sh = int(torch.randint(-15, 16, (1,), generator=g)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + 0.2 * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g).to(X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g) > 0.1).float().to(X.device)
    amp = (0.8 + 0.45 * torch.rand(Xs.shape[0], 1, 1, generator=g)).to(X.device); return Xs * keep * amp


def ntxent(z1, z2, t=0.1):
    z = F.normalize(torch.cat([z1, z2]), dim=1); n = len(z1); S = z @ z.T / t
    S = S.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device); return F.cross_entropy(S, tgt)


def pretrain(m, d, RX, graz, rows, seed, steps, check=True):
    st = d["stream"][rows]; X = []
    for s in dict.fromkeys(st):
        r = rows[st == s]; X.append(m._aligned(r, False)); rg = r[graz[r]]
        if len(rg):
            X.append(m._aligned(rg, False, RX))
    X = torch.from_numpy(np.concatenate(X)).to(m.dev); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = deep.build(m.arch, X.shape[1], X.shape[2], m.pretrained, **{**m.net_kw, "n_outputs": 64}).to(m.dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); t0 = time.time(); first = last = None
    for step in range(steps):
        b = torch.randint(0, len(X), (256,), generator=g).to(m.dev); xb = X[b]
        loss = ntxent(net(aug(xb, g)), net(aug(xb, g))); opt.zero_grad(); loss.backward(); opt.step()
        first = float(loss) if first is None else first; last = float(loss)
    print(f"   ssl pretrain: {len(X)} windows ({int(graz[rows].sum())} rest), {steps} steps, loss {first:.3f} -> {last:.3f}, {time.time() - t0:.0f}s", flush=True)
    assert last < first or not check, "the contrastive loss did not fall"
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def run(d, seed, smoke=False):
    RX = np.load(WS / "data/proteus_windows_start-4.5.npz")["X"]; assert len(RX) == len(d["y"])
    graz = np.char.endswith(d["task"].astype(str), "Graz"); OUT.mkdir(parents=True, exist_ok=True); PRE.mkdir(parents=True, exist_ok=True)
    res = {"ssl": [], "twin": []}; idx = []
    for fold, tr, te in protocol.folds(d, "all_loso"):
        rows = np.flatnonzero(tr); f = PRE / f"{'smoke-' if smoke else ''}fblight_ea-ssl_{fold}-s{seed}.pt"  # a smoke never leaves a checkpoint the real run would reuse
        m = SB.make(CFG, d, seed)
        if smoke:
            m.epochs = 1
        if not f.exists():
            torch.save({"state": pretrain(m, d, RX, graz, rows, seed, 20 if smoke else 1500, check=not smoke)}, f)
        for arm in ("twin", "ssl"):
            m = SB.make(CFG, d, seed); m.init = str(f) if arm == "ssl" else m.init
            if smoke:
                m.epochs = 1
            m.fit(rows); m.set_k(K); P = protocol.predict_streams(m, d, te, proba=True); res[arm].append(P)
            assert len(P) == int(te.sum()), (len(P), int(te.sum()))
        idx.append(np.flatnonzero(te)); print(f"{fold} s{seed}: twin {np.mean(res['twin'][-1].argmax(1) == d['y'][idx[-1]]):.3f} ssl {np.mean(res['ssl'][-1].argmax(1) == d['y'][idx[-1]]):.3f}", flush=True)
        if smoke:
            return print("SMOKE OK", flush=True)
    I = np.concatenate(idx); o = np.argsort(I)
    for arm, Ps in res.items():
        P = np.concatenate(Ps)[o]; assert np.isfinite(P).all() and len(P) == len(d["y"])
        np.savez(OUT / f"ssl_fbl_{arm}-all_loso-s{seed}.npz", proba=P, index=I[o]); print(f"TOTAL {arm} s{seed} acc {np.mean(P.argmax(1) == d['y'][I[o]]):.4f}", flush=True)


if __name__ == "__main__":
    smoke = sys.argv[1] == "smoke"; run(protocol.load(), 0 if smoke else int(sys.argv[1]), smoke)
