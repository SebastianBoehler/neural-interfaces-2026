"""Lane K test 2: one raw FBCNet SessionBN member (seed SEED, N9 = all other people, B5 recipe) streamed on the held person in the pinned full0
order, exporting the four segment logit contributions of its linear head.
python seg_export.py SEED OUT_DIR   (run as ROOT/laneK/seg_export.py from a B5-style release dir). Writes OUT_DIR/seg_{seed}_{held}.npz for
every held person P01-P10: delivery rows, posterior (n x 3), seg (n x 4 x 3) with seg.sum(1) + bias == logits (asserted)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
import session_bn  # noqa: E402

ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
PEOPLE = [f"P{i:02}" for i in range(1, 11)]
STRIDE = 4


def main(seed, out):
    d = protocol.load(())
    perms = ROOT / "runs/shuffle_perms.npz"
    assert hashlib.sha256(perms.read_bytes()).hexdigest() == ORDER_SHA
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    with np.load(perms) as p:
        order = p["full|0|all_loso"]
    for held in PEOPLE:
        query = np.flatnonzero(person == held)
        rows = order[np.isin(order, query)]
        train = np.flatnonzero(person != held)
        run_cv.SEED[0] = seed
        net = session_bn.make("fbcnet_raw_sbn", d, seed)
        assert net.init is None and net.epochs == 40
        net.fit(train)
        net.set_k(20.0)
        net.net.eval()
        lin = [m for m in net.net.final_layer.modules() if isinstance(m, nn.Linear)][-1]
        cap = {}
        h = lin.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach().double().cpu()))
        post, seg, logit = [], [], []
        for st in dict.fromkeys(d["stream"][rows]):
            r = rows[d["stream"][rows] == st]
            pr = net.predict_proba_stream(r)
            x = cap["x"]  # (n, F*STRIDE), feature-major then segment
            assert x.shape[1] % STRIDE == 0, x.shape
            W = lin.weight.detach().double().cpu().view(3, -1, STRIDE)  # class, F, seg
            c = torch.einsum("nfs,kfs->nsk", x.view(len(r), -1, STRIDE), W).numpy()  # n x seg x class
            lg = c.sum(1) + lin.bias.detach().double().cpu().numpy()
            np.testing.assert_allclose(torch.softmax(torch.from_numpy(lg), 1).numpy(), pr, atol=1e-4)
            post.append((r, pr)); seg.append(c)
        h.remove()
        r = np.concatenate([a for a, _ in post]); pr = np.concatenate([b for _, b in post]); c = np.concatenate(seg)
        np.savez(Path(out) / f"seg_{seed}_{held}.npz", rows=r, post=pr, seg=c, bias=lin.bias.detach().cpu().numpy(), delivery=rows)
        print("PASS", held, seed, len(r), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
