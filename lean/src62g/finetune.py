"""v6 in-session fine-tuning of one SessionBN trace per family (gC5 delta1; ledger §113, §152, §120)."""
import copy
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STATE = ("s1", "s2", "n")
# v6.2g: the process-level wall-clock budget lives here, not on the model (the evaluator deep-copies the model per session)
FT_CLOCK = {"t0": None, "stopped": False}


def ft_clock_start():
    """Called once by Solver.load_model: the fine-tuning budget counts from here."""
    FT_CLOCK["t0"], FT_CLOCK["stopped"] = time.monotonic(), False


def strong(X, g, shift=15, noise=0.2, drop=0.1):
    """studies/13nov/ttt7.strong_dev: a time shift, Gaussian noise at a share of each channel's sd and a share of
    channels zeroed, every draw from the generator on X's device"""
    sh = int(torch.randint(-shift, shift + 1, (1,), generator=g, device=X.device))
    Xs = torch.roll(X, sh, -1)
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g, device=X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g, device=X.device) > drop).float()
    return Xs * keep


class FineTune:
    """Per family (``v6_slot_<name>``: its member indices), the trace of member ``v6_ft_member`` is fine-tuned in
    the session from its shipped weights (one copy carried through the session: warm start). At window t >= t0 with
    (t - t0) % every == 0: a new Adam (lr), ``steps`` steps on windows 0..t-1 of the member's own input stream, each
    = forward without grad from zero StreamBN state, forward on a strong augmentation, CE on the labels (argmax of
    the package's own earlier outputs, weighted by their maximum) + KL(clean || augmented), backward, step; then one
    pass over windows 0..t-1 to rebuild the StreamBN state. From t0 on, every window is predicted from that state.
    Trained: every parameter the trace exposes with requires_grad (the FIR banks are frozen) and the StreamBN affine."""

    def __init__(self, z, w, dev, root):
        names = [str(s) for s in z["v6_slots"]]
        self.slots = {s: z[f"v6_slot_{s}"].astype(int) for s in names}
        self.member = dict(zip(names, z["v6_ft_member"].astype(int)))
        self.t0, self.every, self.n_steps = int(z["v6_ft_t0"]), int(z["v6_ft_every"]), int(z["v6_ft_steps"])
        self.lr, self.lam, self.seed = float(z["v6_ft_lr"]), float(z["v6_ft_lam"]), int(z["v6_ft_seed"])
        self.on = bool(int(z["v6_ft_on"]))  # 0: no fine-tuning (the v6 chain keeps the eye-net view and the rest member)
        # v6.2g guards: no refit once the prefix exceeds v6_ft_cap windows (0: none); no refit and frozen nets once the
        # process has run v6_ft_budget_s seconds since load_model (0: none). Inert below both thresholds.
        self.cap, self.budget = int(z["v6_ft_cap"]), float(z["v6_ft_budget_s"])
        self.files = {s: str(z[f"net{self.member[s]}_sjit_file"]) for s in names}
        # session copies are loaded from the files' bytes: a deep-copied TorchScript module cannot be optimised
        self.blobs = {s: (Path(root) / f).read_bytes() for s, f in self.files.items()}
        self.dev = dev
        for s in names:
            assert self.member[s] in self.slots[s] and all(w[1 + i] > 0 for i in self.slots[s]), s
        self.reset_state()

    def __deepcopy__(self, memo):  # the session's modules are made in reset_state
        new = copy.copy(self)
        new.reset_state()
        return new

    def reset_state(self):
        self.nets = {s: torch.jit.load(io.BytesIO(b), map_location=self.dev) for s, b in self.blobs.items()}
        self.theta0 = {s: [p.detach().clone() for p in self.params(m)] for s, m in self.nets.items()}
        self.gens = {s: torch.Generator(device=self.dev).manual_seed(self.seed) for s in self.nets}
        self.X, self.lab, self.wt = {s: [] for s in self.nets}, [], []
        self.refits, self.steps, self.dtheta = 0, 0, 0.0
        self.capped, self.frozen = 0, 0  # refits skipped by the cap; windows on the frozen nets after the budget

    @staticmethod
    def params(m):
        ps = [p for p in m.parameters() if p.requires_grad]
        return ps + [b for n, b in m.named_buffers() if n.rsplit(".", 1)[-1] in ("w", "b")]

    @staticmethod
    def zero(m):
        for n, b in m.named_buffers():
            if n.rsplit(".", 1)[-1] in STATE:
                b.detach_()
                b.zero_()

    def refit(self, t):
        # the profiling executor would re-specialise the traces for every new prefix length t: run unoptimised here
        # (per window, at a fixed batch of 1, the traces keep the default executor)
        with torch.jit.optimized_execution(False):
            self._refit(t)

    def _refit(self, t):
        lab = torch.as_tensor(np.asarray(self.lab[:t]), device=self.dev)
        wt = torch.as_tensor(np.asarray(self.wt[:t], dtype=np.float32), device=self.dev)
        for s, m in self.nets.items():
            X = torch.from_numpy(np.stack(self.X[s][:t])).to(self.dev)
            ps = self.params(m)
            for p in ps:
                p.requires_grad_(True)
            opt = torch.optim.Adam(ps, lr=self.lr)
            for _ in range(self.n_steps):
                opt.zero_grad()
                self.zero(m)
                with torch.no_grad():
                    zc = m(X)[:, :3]
                self.zero(m)
                za = m(strong(X, self.gens[s]))[:, :3]
                ce = (wt * F.cross_entropy(za, lab, reduction="none")).sum() / wt.sum().clamp_min(1e-6)
                loss = ce + self.lam * F.kl_div(F.log_softmax(za, 1), F.softmax(zc, 1), reduction="batchmean")
                loss.backward()
                opt.step()
            self.steps += self.n_steps
            with torch.no_grad():
                self.zero(m)
                m(X)  # StreamBN state over windows 0..t-1
                self.dtheta = max(self.dtheta, max(float((p - q).abs().max()) for p, q in zip(self.params(m), self.theta0[s])))
        self.refits += 1

    def relabel(self, outs):
        """v6.2: replace every stored label by the given outputs (argmax, weight = max), one per window so far."""
        assert len(outs) == len(self.lab), (len(outs), len(self.lab))
        self.lab = [int(np.argmax(o)) for o in outs]
        self.wt = [float(np.max(o)) for o in outs]

    def step(self, members, inputs, out):
        """members: {index: frozen posterior}; inputs: {index: this window's network input}; out: the package's
        output of this window (a label for later refits). -> {family: fine-tuned posterior of this window} from t0 on."""
        t, res = len(self.lab), {}
        if not self.on:
            self.lab.append(int(np.argmax(out)))
            return res
        if self.budget and not FT_CLOCK["stopped"] and FT_CLOCK["t0"] is not None and time.monotonic() - FT_CLOCK["t0"] > self.budget:
            FT_CLOCK["stopped"] = True
            print(f"[v6-ft] wall-clock budget of {self.budget:.0f} s reached: no further fine-tuning in this process "
                  "(frozen nets from here on)", file=sys.stderr, flush=True)
        if self.budget and FT_CLOCK["stopped"]:  # frozen nets for the rest of the process
            self.frozen += 1
            self.lab.append(int(np.argmax(out)))
            self.wt.append(float(np.max(out)))
            return res
        for s in self.nets:
            self.X[s].append(np.asarray(inputs[self.member[s]], dtype=np.float32))
        if t >= self.t0:
            if (t - self.t0) % self.every == 0:
                if self.cap and t > self.cap:  # long session: no new refit; the last fine-tuned nets keep predicting
                    self.capped += 1
                else:
                    self.refit(t)
            with torch.no_grad():
                for s, m in self.nets.items():
                    x = torch.from_numpy(self.X[s][t][None]).to(self.dev)
                    res[s] = torch.softmax(m(x)[:, :3].double(), 1)[0].cpu().numpy()
        self.lab.append(int(np.argmax(out)))
        self.wt.append(float(np.max(out)))
        return res
