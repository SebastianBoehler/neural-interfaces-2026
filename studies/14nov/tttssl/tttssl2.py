"""TTT-SSL wave 2 (ledger s255 amendment): tttssl.py with masked reconstruction and lower auxiliary weights.
Forms: R10 / R25 masked reconstruction at aux weight 0.1 / 0.25; C10 SimCLR (tttssl's C) at 0.1. Reconstruction: one
contiguous time block of U(25 %, 40 %) of the samples and U(25 %, 40 %) of the channels zeroed per window; the encoder's
features (input of final_layer) -> decoder 32 -> 128 ELU -> 9 x 41 -> the clean window's log band power in the net's own
filter bank, z-scored per dimension over the batch (detached); MSE = 1 - R2. Arms: na, sB40 (primary), sA10 (secondary).
Twin: wave 1's tttssl_twin_na files. Outputs runs/tttssl/probs/tttssl_<form>_<arm>-s<seed>.npz.
Usage (Galvani): python studies/14nov/tttssl/tttssl2.py TASK (form = FORMS[TASK % 3], seed = TASK // 3) | smoke"""
import importlib.util, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
_s = importlib.util.spec_from_file_location("tttssl", Path(__file__).resolve().parent / "tttssl.py")
T = importlib.util.module_from_spec(_s); _s.loader.exec_module(T)

FORMS = ("R10", "R25", "C10"); LAMS = {"R10": 0.1, "R25": 0.25, "C10": 0.1}
ARMS2 = {"na": None, "sB40": ("ssl", "bn", 5e-3, 40), "sA10": ("ssl", "all", 1e-3, 10)}
_aux1 = T.aux_loss


def mask(X, g):
    n, C, L = X.shape; dev = X.device
    fr = 0.25 + 0.15 * torch.rand(n, 1, generator=g, device=dev); ln = (fr * L).long()
    st = (torch.rand(n, 1, generator=g, device=dev) * (L - ln + 1)).long(); t = torch.arange(L, device=dev)[None]
    keep_t = ~((t >= st) & (t < st + ln))
    fc = 0.25 + 0.15 * torch.rand(n, 1, generator=g, device=dev); keep_c = torch.rand(n, C, generator=g, device=dev) >= fc
    return X * keep_t[:, None, :].float() * keep_c[:, :, None].float()


def target(net, X):
    with torch.no_grad():
        Y = torch.log(net.spectral_filtering(X).var(-1) + 1e-6).flatten(1)  # (n, bands x channels)
        return (Y - Y.mean(0)) / Y.std(0).clamp_min(1e-6)


def aux_loss(net, X, form, g, mir, acc=False):
    if form.startswith("R"):
        net.final_layer._want = True; net(mask(X, g)); net.final_layer._want = False
        f = net.final_layer._feat; del net.final_layer._feat  # no graph left on the module (deepcopy)
        return F.mse_loss(net.tttdec(f), target(net, X))
    return _aux1(net, X, form[0], g, mir, acc)


def _hook(mod, inp, out):
    if getattr(mod, "_want", False):
        mod._feat = inp[0]


class RNet(T.JointNet):
    def _train(self, *a, **k):
        net = self.net
        if not hasattr(net, "tttdec"):
            nb = net.spectral_filtering(torch.zeros(1, 41, 480, device=next(net.parameters()).device)).shape[1]
            torch.manual_seed(self.seed + 11)
            net.add_module("tttdec", nn.Sequential(nn.Linear(net.final_layer.in_features, 128), nn.ELU(), nn.Linear(128, nb * 41)).to(self.dev))
            net.final_layer.register_forward_hook(_hook)
        n0 = sum(p.numel() for p in net.parameters()); super()._train(*a, **k)
        assert n0 == sum(p.numel() for p in net.parameters()) and hasattr(net, "tttdec")


def make(form, d, seed, mir):
    m = T.SB.make(T.CFG, d, seed); assert type(m) is T.SB.SessionNet and not m.aug and m.init is None and not m.net_kw
    m.__class__ = RNet if form.startswith("R") else T.JointNet; m.form, m.mir = form, torch.from_numpy(mir)
    m.net_kw = {"n_outputs": T.NOUT[form[0]] if form[0] != "R" else 3}; return m


T.aux_loss, T.make, T.FORMS = aux_loss, make, FORMS
T.ARMS.update({f: ARMS2 for f in FORMS})
assert T.run.__globals__["aux_loss"] is aux_loss and T.JointNet._train.__globals__["make"] is make


def main():
    d = T.protocol.load(); T.OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "smoke":
        (T.OUT.parent / "smoke2").mkdir(exist_ok=True)
        for form in FORMS:
            T.LAM = LAMS[form]; T.run(form, d, 0, T.OUT.parent / "smoke2", smoke=True)
        return
    task = int(sys.argv[1]); form = FORMS[task % 3]; T.LAM = LAMS[form]; print(f"form {form} LAM {T.LAM}", flush=True)
    assert T.JointNet._train.__globals__["LAM"] == LAMS[form]; T.run(form, d, task // 3, T.OUT)


if __name__ == "__main__":
    main()
