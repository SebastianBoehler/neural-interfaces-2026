"""Lane P2: segment contributions of the REAL package member, leave-one-person-out.

usage: seg_member.py HELD OUT PKG_ZIP_OR_SUBMISSION_PY   (run as ROOT/laneP2/seg_member.py from a release dir that carries experiments/, 2026-competition/, data/)
The member is the cap250 raw FBCNet SessionBN seed 0 (package net 47, sbn_fbcnet_raw_k20_s0.pt): same recipe as laneP/seg_export.py (fbcnet_raw_sbn, seed 0,
N9 = all people but HELD, k 20), but here the fit is exported as a stateful TorchScript trace (copy of cleanslate/cap250/families/_sbn.py export) and the held
person's two pinned orders are streamed window by window through the PACKAGE class `SharedFIRSJit` (code of PKG_SUBMISSION, with its segment hook), reset at
every session. Checks, all hard asserts: package posterior vs eager SessionBN < 1e-4; hook segments + bias recover the logits (the hook returns None otherwise);
the trace is stateful with zero state. Writes OUT/seg_{full0|within0}_0_{HELD}.npz (rows, post = package posterior, seg n x 4 x 3, bias, delivery, post_eager)."""
import hashlib
import os
import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding"), str(ROOT / "2026-competition")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
import session_bn  # noqa: E402
from session_bn import STATE, reset_stream, to_stream  # noqa: E402

ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
NAME = "sbn_fbcnet_raw_k20_s0.pt"


def device_free_trace(module, example):  # copy of cleanslate/cap250/families/_sbn.py
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


def main(held, out, pkg_source):
    d = protocol.load(())
    perms = ROOT / "runs/shuffle_perms.npz"
    assert hashlib.sha256(perms.read_bytes()).hexdigest() == ORDER_SHA
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    with np.load(perms) as p:
        orders = {"full0": p["full|0|all_loso"], "within0": p["within|0|all_loso"]}
    query, train = np.flatnonzero(person == held), np.flatnonzero(person != held)
    run_cv.SEED[0] = 0
    net = session_bn.make("fbcnet_raw_sbn", d, 0)
    assert net.init is None and net.epochs == 40
    if os.environ.get("LANEP2_SMOKE") == "1":  # plumbing smoke: 2 epochs, results not used
        net.epochs = 2
    net.fit(train)
    net.set_k(20.0)
    net.net.eval()
    work = {}  # eager reference and aligned inputs on the fit device, before the net moves to CPU for the trace
    for tag, order in orders.items():
        rows = order[np.isin(order, query)]
        for st in dict.fromkeys(d["stream"][rows]):
            r = rows[d["stream"][rows] == st]
            work[tag, st] = (r, net.predict_proba_stream(r), net._aligned(r, causal=True))
    net.net.cpu().eval()
    net.dev = "cpu"
    first = work[next(iter(work))][2]
    traced = device_free_trace(to_stream(net.net, 20.0), torch.from_numpy(first[:4]))
    reset_stream(traced)
    with TemporaryDirectory() as tmp:
        if pkg_source.endswith(".zip"):  # the package's own submission.py
            (Path(tmp) / "pkg").mkdir()
            (Path(tmp) / "pkg/submission.py").write_bytes(ZipFile(pkg_source).read("submission.py"))
            pkg_source = str(Path(tmp) / "pkg/submission.py")
        spec = importlib.util.spec_from_file_location("p2_pkg", pkg_source)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        traced.save(str(Path(tmp) / NAME))
        member = mod.SharedFIRSJit({"sjit_file": np.array(NAME)}, "cpu", tmp)
        buffers = [b for n, b in member.m.named_buffers() if n.rsplit(".", 1)[-1] in STATE]
        assert buffers and all(float(b.abs().sum()) == 0 for b in buffers), "trace not stateful with zero state"
        member.seg_on = True
        bias = None
        for tag in orders:
            parts = {k: v for k, v in work.items() if k[0] == tag}
            rows, post, seg, eager = [], [], [], []
            for (_, st), (r, pe, H) in parts.items():
                member.reset_state()
                for t in range(len(r)):
                    p = member.proba(torch.from_numpy(H[t:t + 1]).to(torch.float32), {}).numpy()[0]
                    assert member.seg is not None, "segment hook did not recover the logits"
                    post.append(p); seg.append(member.seg); member.seg = None
                rows.append(r); eager.append(pe)
            lin = [m for m in member.m.final_layer.modules() if m.original_name == "Linear"][-1]
            bias = lin.bias.detach().double().cpu().numpy()
            r, post, seg, eager = np.concatenate(rows), np.stack(post), np.stack(seg), np.concatenate(eager)
            diff = float(np.abs(post - eager).max())
            assert diff < 1e-4, (tag, diff)
            np.savez(Path(out) / f"seg_{tag}_0_{held}.npz", rows=r, post=post, seg=seg, bias=bias, delivery=orders[tag][np.isin(orders[tag], query)], post_eager=eager)
            print("PASS", tag, held, len(r), "pkg-vs-eager max|diff|", diff, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
