"""RESTBLOCKS (ledger s207, in-net edit of the rest net): restshuf.py with a hook on the net's final convolution that saves,
per window, every filter's logit contribution z_f (first three outputs) and the bias, in recording order and in one
delivery order. A per-filter gain edit z = sum_f g_f z_f + b is then exact on CPU (it is a gain on the final layer's
weights per input filter). Asserts the contributions reproduce the net's posteriors (1e-5). New people only (all_loso).
Saved to runs/diag/restblocks/<order>/{rec,dlv}/rest4_sh_r1-all_loso-s<seed>.npz (index, proba, contrib (N, F, 3),
bias (N, 3)); each order keeps its own recording-order copy, so parallel tasks never share a file.
Usage (Leonardo): SCHEME=full PSEED=0 python studies/14nov/restblocks.py SEED | smoke   (smoke: 1 epoch, first fold)"""
import importlib.util, os, sys
from pathlib import Path
import numpy as np
import torch
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); SCHEME = os.environ["SCHEME"]; PSEED = int(os.environ["PSEED"])
_spec = importlib.util.spec_from_file_location("rest4", WS / "studies/11nov/rest4/rest4.py"); R4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R4)
sys.path.insert(0, str(WS / "studies/13nov"))
import deliver  # noqa: E402
PERMS = np.load(WS / "runs/shuffle_perms.npz"); CAP = {}
_orig = R4.Rest4.predict_proba_stream


def _stream(self, idx):  # one forward per stream: the hook sees the stream's windows in the order of idx
    conv = [m for m in self.net.modules() if isinstance(m, torch.nn.Conv2d)][-1]; assert conv.out_channels == self.nout, conv
    box = {}
    def grab(mod, inp, out):  # must return None: a forward hook that returns a value replaces the module's output
        box["x"] = inp[0].detach()
    h = conv.register_forward_hook(grab)
    try:
        assert not getattr(self, "tta", ()), "TTA would mix shifted forwards"
        P = _orig(self, idx)
    finally:
        h.remove()
    W, b = conv.weight.detach()[:3].double(), conv.bias.detach()[:3].double()  # (3, F, kt, kw)
    assert conv.dilation == (1, 1) and W.shape[-1] == 1, (conv.dilation, W.shape)
    xu = box["x"].double()[..., 0].unfold(2, W.shape[2], conv.stride[0])  # (B, F, T', kt): the conv slides over time
    zft = torch.einsum("bftk,cfk->bfct", xu, W[..., 0]).cpu().numpy(); bb = b.cpu().numpy()  # per filter and time step
    best = None
    for name, red in (("mean", lambda a: a.mean(-1)), ("sum", lambda a: a.sum(-1))):  # how the net reduces the time steps
        zf = red(zft); q = zf.sum(1) + bb; q = np.exp(q - q.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        if np.abs(q - P).max() < 1e-5:
            best = name; break
    assert best, f"no linear reduction over time reproduces the net (mean/sum); last error {np.abs(q - P).max():.2e}"
    for i, r in enumerate(idx):
        CAP[int(r)] = (zf[i], bb)
    return P


R4.Rest4.predict_proba_stream = _stream


def main():
    smoke = sys.argv[1] == "smoke"; seed = 0 if smoke else int(sys.argv[1]); d, d2, sel = R4.data(); proto = "all_loso"
    for order in ("rec", "dlv"):
        out = WS / "runs/diag/restblocks" / f"{SCHEME}{PSEED}{'_smoke' if smoke else ''}" / order; out.mkdir(parents=True, exist_ok=True); CAP.clear()
        if order != "rec":
            deliver.install(R4.protocol, deliver.rank_from(PERMS, SCHEME, PSEED, proto))
        try:
            R4.run("sh", "r1", d, d2, sel, proto, seed, out, epochs=1 if smoke else 0)
        finally:
            if order != "rec":
                deliver.uninstall(R4.protocol)
        f = out / f"rest4_sh_r1-{proto}-s{seed}.npz"; z = dict(np.load(f)); idx = z["index"]
        if "contrib" in z:
            continue
        assert all(int(r) in CAP for r in idx), "windows without captured contributions"
        np.savez(f, **z, contrib=np.stack([CAP[int(r)][0] for r in idx]).astype(np.float32), bias=np.stack([CAP[int(r)][1] for r in idx]))
        print(f"saved {f} with filter contributions {np.stack([CAP[int(r)][0] for r in idx]).shape}", flush=True)
    if smoke:
        print("SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()
