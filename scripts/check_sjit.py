"""Unit check of the session-stateful TorchScript member (submission.py SJit + experiments/session_bn.py).

Usage: python scripts/check_sjit.py [--cuda]   (random nets, seconds)
For FBCNet, ShallowFBCSPNet, FBLightConvNet and SincShallowNet (the last two: REC, 6 Nov) with SessionBN
layers (random running statistics, test prior k 5): the traced StreamBN module, loaded through submission.SJit, streamed window by window equals the batch-causal SessionBN
evaluation; reset_state restarts a fresh session; a deepcopy carries independent state; one batch call equals
the window-by-window stream; exactly 3 state buffers per batch norm, all zero after export.
"""
import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

import braindecode.models as bd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts"), str(ROOT / "2026-competition/tracks/bci_decoding")]
from export_external_teacher import device_free_trace  # noqa: E402
from session_bn import STATE, SessionBN, StreamBN, reset_stream, swap_bn, to_stream  # noqa: E402


def submission():
    spec = importlib.util.spec_from_file_location("sub_sjit", ROOT / "submissions/ensemble/submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    dev = "cuda" if "--cuda" in sys.argv else "cpu"
    sub = submission()
    torch.manual_seed(0)
    ok = True
    for arch, kw in (("FBCNet", {"sfreq": 120.0}), ("ShallowFBCSPNet", {}), ("FBLightConvNet", {"sfreq": 120.0}),
                     ("SincShallowNet", {"sfreq": 120.0})):
        net = getattr(bd, arch)(n_chans=41, n_outputs=3, n_times=480, **kw)
        nbn = swap_bn(net, False, 20.0)
        for m in net.modules():
            if isinstance(m, SessionBN):
                m.running_mean.normal_(0, 0.3)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.data.uniform_(0.5, 1.5)
                m.bias.data.normal_(0, 0.2)
                m.k = 5.0
        net.eval()
        A, B = torch.randn(9, 41, 480), torch.randn(7, 41, 480) * 1.5 + 0.1
        with torch.no_grad():
            ref = {n: torch.softmax(net(X).double(), 1).numpy() for n, X in (("A", A), ("B", B))}
            sm = to_stream(net, 5.0)
            tr = device_free_trace(sm, A[:4])
        reset_stream(tr)
        with tempfile.TemporaryDirectory() as tmp:
            tr.save(f"{tmp}/m.pt")
            sj = sub.SJit({"sjit_file": np.array("m.pt")}, dev, tmp)
        bufs = {n: b for n, b in sj.m.named_buffers() if n.rsplit(".", 1)[-1] in STATE}
        n_stream = sum(isinstance(m, StreamBN) for m in sm.modules())
        rows = lambda m, X: np.concatenate([m.proba(X[i:i + 1].numpy()) for i in range(len(X))])
        res = {"state_buffers": len(bufs) == 3 * nbn == 3 * n_stream,
               "saved_state_zero": all(float(b.abs().sum()) == 0 for b in bufs.values())}
        sj.reset_state()
        got_a = rows(sj, A)
        res["stream_eq_batch_causal"] = float(np.abs(got_a - ref["A"]).max())
        sj.reset_state()
        res["reset_fresh_session"] = float(np.abs(rows(sj, B) - ref["B"]).max())
        sj.reset_state()
        res["one_batch_call_eq_stream"] = float(np.abs(sj.proba(A.numpy()) - got_a).max())
        sj.reset_state()
        head = rows(sj, A[:4])
        cp = copy.deepcopy(sj)
        cp.reset_state()
        rows(cp, B)  # advances only the copy
        tail = rows(sj, A[4:])
        res["deepcopy_independent"] = float(np.abs(np.r_[head, tail] - ref["A"]).max())
        res["copy_has_own_module"] = cp.m is not sj.m
        print(arch, dev, res, flush=True)
        ok &= res["state_buffers"] and res["saved_state_zero"] and res["copy_has_own_module"] and all(
            v < 1e-5 for k, v in res.items() if isinstance(v, float))
    print("SJIT CHECK", "PASS" if ok else "FAIL", flush=True)
    assert ok


if __name__ == "__main__":
    main()
