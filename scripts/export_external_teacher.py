"""Add the external-head teacher (MI-transfer arm, 1 Nov) to a package as a TorchScript member.

Usage: python scripts/export_external_teacher.py SRC_DIR OUT_DIR [--weight 2] [--ckpt-dir runs/pretrain]
       [--families sincshallow,fbcnet] [--seeds 0-5] [--C 0.1]
The teacher (docs/SEALED_PLAN.md, "External-head teacher"): the 12 external-only pretrained
nets (sincshallow_ea-m7r and fbcnet_ea-m7r, seeds 0-5; seven MI corpora, no PROTEUS data)
are frozen; their per-corpus head log-probs are averaged over seeds per family (2 x 19), and a
logistic map (C 0.1, balanced) fit on all public windows turns them into 3-class logits. Its
input is the package's causal EA stream (alignment keys copied from an EA member), so it adds
no session state. It enters the member mix with weight WEIGHT x the v2 weight (one family).
Checks on all public windows: traced vs eager teacher (batch 256 and 1), and the streamed
package against the harness reference (old mix + teacher, rebalancing, R19e stage).
Writes OUT_DIR/teacher.pt, weights.npz, submission.py and OUT_DIR_check.json.
The defaults reproduce the packaged arm; the 2 Nov candidate (docs/OVERNIGHT_2NOV.md) is
--families sincshallow,fbcnet,fblight,shallow --seeds 0-11 --C 0.01 --weight 4.
"""

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path

import braindecode.models as bd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import session_em  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from deep import ARCHS  # noqa: E402
from ensemble import rebalance  # noqa: E402

ALIGN = ("align", "k", "G", "scale", "shrink")


class Heads(nn.Module):
    """Frozen external nets -> (B, families x 19) seed-averaged per-corpus head log-probs."""

    def __init__(self, ckpt_dir, seeds, families):
        super().__init__()
        self.fams = nn.ModuleList()
        self.blocks = []
        for arch in families:
            nets = nn.ModuleList()
            for s in seeds:
                ck = torch.load(Path(ckpt_dir) / f"{arch}_ea-m7r-s{s}.pt", weights_only=False)
                corp = json.loads(ck["meta"])["corpora"]
                blocks = [(m["offset"], m["offset"] + len(m["classes"])) for m in corp.values()]
                assert not self.blocks or self.blocks == blocks
                self.blocks = blocks
                cls = getattr(bd, ARCHS[arch])
                kw = dict(n_chans=41, n_outputs=blocks[-1][1], n_times=480)
                if "sfreq" in inspect.signature(cls).parameters:
                    kw["sfreq"] = 120.0
                net = cls(**kw)
                net.load_state_dict(ck["state"])
                # metadata only (forward unchanged): unset braindecode properties raise
                # ValueError, which breaks torch.jit.trace's hasattr scan of the module
                net._chs_info = net._chs_info or [{"ch_name": f"ch{c}"} for c in range(41)]
                net._sfreq = net._sfreq or 120.0
                net._input_window_seconds = net._input_window_seconds or 4.0
                nets.append(net.eval())
            self.fams.append(nets)

    def forward(self, x):
        out = []
        for nets in self.fams:
            acc = []
            for net in nets:
                z = net(x)
                acc.append(torch.cat([torch.log_softmax(z[:, a:b], 1) for a, b in self.blocks], 1))
            out.append(torch.stack(acc).mean(0))
        return torch.cat(out, 1)


class Teacher(nn.Module):
    def __init__(self, heads, coef, intercept):
        super().__init__()
        self.heads = heads
        self.lin = nn.Linear(coef.shape[1], 3)
        with torch.no_grad():
            self.lin.weight.copy_(torch.from_numpy(coef).float())
            self.lin.bias.copy_(torch.from_numpy(intercept).float())

    def forward(self, x):
        return self.lin(self.heads(x))


def device_free_trace(module, example):
    """Trace with same-device ``.to(device)`` calls skipped: braindecode's ``.to(x.device)``
    (sinc filters, FBCNet filter bank) would otherwise bake a CPU constant into the graph."""
    to = torch.Tensor.to

    def same_device_noop(self, *args, **kw):
        if not kw and len(args) == 1 and isinstance(args[0], torch.device) and args[0] == self.device:
            return self
        return to(self, *args, **kw)

    torch.Tensor.to = same_device_noop
    try:
        with torch.no_grad():
            traced = torch.jit.trace(module, example, check_trace=False)
    finally:
        torch.Tensor.to = to
    assert 'value="cpu"' not in traced.inlined_graph.str(), "device constant left in the trace"
    return traced


def batched(f, X, n=256):
    with torch.no_grad():
        return torch.cat([f(torch.from_numpy(X[i:i + n])) for i in range(0, len(X), n)]).double().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--ckpt-dir", default=str(ROOT / "runs/pretrain"))
    ap.add_argument("--families", default="sincshallow,fbcnet")
    ap.add_argument("--seeds", default="0-5")
    ap.add_argument("--C", type=float, default=0.1)
    a = ap.parse_args()
    torch.set_num_threads(1)
    with np.load(a.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    n = int(z["n_nets"])
    j = next(i for i in range(n) if bool(z[f"net{i}_align"]) and f"net{i}_jit_file" not in z)
    p = {k: z[f"net{j}_{k}"] for k in ALIGN}

    d = protocol.load()
    st = d["stream"]
    mod = load_solver(a.src, "src_pkg")  # reuse the package's own InputStream
    InputStream = type(next(iter(mod.groups.values()))[0])
    H, ins = np.empty(d["X"].shape, np.float32), InputStream(p)
    for i in range(len(H)):
        if i == 0 or st[i] != st[i - 1]:
            ins.reset_state()
        H[i] = ins(d["X"][i].astype(np.float32))
    lo, _, hi = a.seeds.partition("-")
    heads = Heads(a.ckpt_dir, range(int(lo), int(hi or lo) + 1), a.families.split(",")).eval()
    Fh = batched(heads, H)
    lr = LogisticRegression(C=a.C, max_iter=3000, class_weight="balanced").fit(Fh, d["y"])
    teacher = Teacher(heads, lr.coef_, lr.intercept_).eval()
    eager = batched(teacher, H)
    traced = device_free_trace(teacher, torch.from_numpy(H[:4]))
    a.out.mkdir(parents=True, exist_ok=True)
    traced.save(str(a.out / "teacher.pt"))
    loaded = torch.jit.load(str(a.out / "teacher.pt"))
    tr256 = batched(loaded, H)
    tr1 = batched(loaded, H[:64], n=1)
    cuda_diff = None
    if torch.cuda.is_available():  # the worker loads it with map_location="cuda"
        on_gpu = torch.jit.load(str(a.out / "teacher.pt"), map_location="cuda")
        with torch.no_grad():
            g = on_gpu(torch.from_numpy(H[:256]).cuda()).double().cpu().numpy()
        cuda_diff = float(np.abs(g - eager[:256]).max())

    zn = dict(z, n_nets=np.array(n + 1))
    zn.update({f"net{n}_{k}": v for k, v in p.items()})
    zn[f"net{n}_jit_file"] = np.array("teacher.pt")
    w_t = a.weight * float(z["member_weights"][0])
    zn["member_weights"] = np.r_[z["member_weights"], w_t]
    np.savez(a.out / "weights.npz", **zn)
    shutil.copyfile(SRC, a.out / "submission.py")

    rows = np.ones(len(d["y"]), bool)
    old_mix = stream(load_solver(a.src, "old_mix"), d, rows, lambda m, x: m.mixed_proba(x))
    T = torch.softmax(torch.from_numpy(tr256), 1).numpy()
    sw = float(z["member_weights"].sum())
    mix = (old_mix * sw + w_t * T) / (sw + w_t)
    Q = np.concatenate([rebalance(mix[st == s], float(z["rb_alpha"]), float(z["rb_k"]), int(z["rb_w"]))
                        for s in dict.fromkeys(st)])
    ch, X64 = list(d["ch_names"]), d["X"].astype(np.float64)
    Fe = np.c_[session_em.gaze_features(X64, ch),
               session_em.event_features(X64, ch, st, z["em_ev_g"])]
    ref = session_em.apply(Fe, Q, st, lam=float(z["em_lam"])).argmax(1)
    new = stream(load_solver(a.out, "new_pkg"), d, rows, lambda m, x: m.predict(x))
    old = stream(load_solver(a.src, "old_pkg"), d, rows, lambda m, x: m.predict(x))
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    res = {"source": str(a.src), "families": a.families, "seeds": a.seeds, "C": a.C,
           "teacher_weight_units": a.weight, "teacher_member_weight": w_t,
           "align_from_member": j, "n_windows": int(len(new)),
           "teacher_in_sample_acc": float(np.mean(eager.argmax(1) == d["y"])),
           "max_abs_logit_traced_vs_eager": float(np.abs(tr256 - eager).max()),
           "max_abs_logit_batch1_vs_eager": float(np.abs(tr1 - eager[:64]).max()),
           "max_abs_logit_cuda_vs_eager": cuda_diff,
           "package_matches_harness": float(np.mean(new == ref)),
           "labels_changed_vs_source": int(np.sum(new != old)),
           "files_sha256": {f.name: sha(f) for f in sorted(a.out.iterdir()) if f.is_file()}}
    (a.out.parent / f"{a.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["max_abs_logit_traced_vs_eager"] < 1e-4 and res["max_abs_logit_batch1_vs_eager"] < 1e-4
    assert res["package_matches_harness"] == 1.0
    assert cuda_diff is None or cuda_diff < 1e-3


if __name__ == "__main__":
    main()
