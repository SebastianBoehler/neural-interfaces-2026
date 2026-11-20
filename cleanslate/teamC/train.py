"""Team C trainer: cross-person ensemble on causally aligned windows -> package (submission.py + weights.pt).

python cleanslate/teamC/train.py --people P01,P02,... --out DIR [--members shallow:0,eegnet:0,fbnet:0] [--epochs 60]

Alignment at training time mimics the sealed stream: every epoch each session is delivered in a fresh random order and
each window is whitened by the shrunk mean covariance of its own delivered prefix (itself included), exactly as
TTTModel.predict does. Batches are session-homogeneous (BN statistics are per session, as the test-time BN refresh).
Augmentation = submission.augment (rotation near identity in whitened space = virtual subjects, shift, scale, noise).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import submission as S  # noqa: E402

ROOT = HERE.parents[1]


def load_ext(names, root):
    """External people already exported in PROTEUS label space (y 0..2, 41 channels, 120 Hz), e.g. scherer2015."""
    Xs, ys, ss = [], [], []
    for nm in names:
        z = np.load(root / nm / "windows.npz")
        assert z["X"].shape[1:] == (41, 480) and set(np.unique(z["y"])) == {0, 1, 2}, nm
        Xs.append(z["X"].astype(np.float32)), ys.append(z["y"])
        ss.append(np.char.add(np.char.add(z["subject"].astype(str), "|"), z["session"].astype(str)))
    return Xs, ys, ss


def load(people, data, extra=(), exclude=()):
    """Training rows of PEOPLE; each extra cue-offset export (same rows, other start) is its own alignment stream."""
    d = np.load(data)
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    keep = np.isin(person, people) & ~np.isin(np.char.add(np.char.add(person, "|"), d["session"]), list(exclude))
    assert set(person[keep]) == set(people), f"missing people {set(people) - set(person[keep])}"
    stream = np.char.add(np.char.add(person[keep], "|"), d["session"][keep])
    Xs, ys, ss = [d["X"][keep].astype(np.float32)], [d["y"][keep]], [stream]
    for e in extra:
        z = np.load(Path(data).with_name(f"proteus_windows_{e}.npz"))
        assert (z["y"] == d["y"]).all() and (z["subject"] == d["subject"]).all() and (z["session"] == d["session"]).all()
        Xs.append(z["X"][keep].astype(np.float32)), ys.append(z["y"][keep]), ss.append(np.char.add(stream, f"|{e}"))
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int64), np.concatenate(ss)


class Aligner:
    """Per-epoch causal alignment of all training windows on the device."""

    def __init__(self, X, stream, dev, shrink):
        self.X = torch.tensor(X, device=dev)
        Xd = self.X.double()
        self.cov = torch.einsum("nct,ndt->ncd", Xd, Xd) / X.shape[-1]
        self.sess = [np.flatnonzero(stream == s) for s in dict.fromkeys(stream)]
        self.dev, self.shrink = dev, shrink

    def epoch(self, rng):
        A = torch.empty_like(self.X)
        batches = []
        for rows in self.sess:
            p = rows[rng.permutation(len(rows))]
            pt = torch.tensor(p, device=self.dev)
            cum = torch.cumsum(self.cov[pt], 0) / torch.arange(1, len(p) + 1, device=self.dev)[:, None, None]
            A[pt] = S.inv_sqrt(cum, self.shrink) @ self.X[pt]
            q = p[rng.permutation(len(p))]
            batches += [q[i:i + 64] for i in range(0, len(q), 64) if len(q[i:i + 64]) >= 16]
        return A, [batches[i] for i in rng.permutation(len(batches))]


def fit_member(arch, seed, al, y, args, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    net = S.build(arch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (len(y) // 48 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    yt = torch.tensor(y, device=dev)
    k = 0
    for ep in range(args.epochs):
        A, batches = al.epoch(rng)
        net.train()
        for b in batches:
            bt = torch.tensor(b, device=dev)
            xb = S.augment(A[bt], g, rot=args.rot) if args.aug else A[bt]
            loss = F.cross_entropy(net(xb), yt[bt], label_smoothing=args.ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if k < steps - 1:
                sched.step()
                k += 1
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", default="shallow:0,eegnet:0,fbnet:0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--shrink", type=float, default=S.DEFAULT_TTT["shrink"])
    ap.add_argument("--data", default=str(ROOT / "data/proteus_windows.npz"))
    ap.add_argument("--exclude", default="", help="streams left out of training, e.g. P04|2,P04|3 (returning dev)")
    ap.add_argument("--ext", default="", help="external corpora as extra training people, e.g. scherer2015")
    ap.add_argument("--extra", default="", help="comma list of cue-offset exports, e.g. start0.5,start2")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    people = sorted(args.people.split(","))
    X, y, stream = load(people, args.data, [e for e in args.extra.split(",") if e],
                       [e for e in args.exclude.split(",") if e])
    if args.ext:
        ex, ey, es = load_ext([e for e in args.ext.split(",") if e], Path(args.data).parent)
        X, y, stream = np.concatenate([X, *ex]), np.concatenate([y, *ey]).astype(np.int64), np.concatenate([stream, *es])
        print("ext", args.ext, "rows", sum(len(v) for v in ey), "streams", len(set(np.concatenate(es))), flush=True)
    al = Aligner(X, stream, dev, args.shrink)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    members, log = [], []
    for spec in args.members.split(","):
        arch, seed = spec.split(":")
        t0 = time.time()
        net = fit_member(arch, int(seed), al, y, args, dev)
        members.append(dict(arch=arch, kw=None, seed=int(seed), state={k: v.cpu() for k, v in net.state_dict().items()}))
        log.append(dict(arch=arch, seed=int(seed), sec=round(time.time() - t0, 1)))
        print("member", log[-1], flush=True)
    ttt = {**S.DEFAULT_TTT, "shrink": args.shrink}
    torch.save(dict(members=members, ttt=ttt, people=people, args=vars(args)), out / "weights.pt")
    shutil.copy(HERE / "submission.py", out / "submission.py")
    json.dump(dict(people=people, streams=sorted({s.split("|")[0] + "|" + s.split("|")[1] for s in stream}), args=vars(args), members=log, rows=len(y)), open(out / "train.json", "w"), indent=1)


if __name__ == "__main__":
    main()
