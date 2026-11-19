"""Stream the CBraMod-C LOPO fits through the package member (CBMember) in a pinned delivery order.

usage: cb_bank.py ROOT CKPT_DIR OUT.npz ORDER SEED [--people P01,...]
ROOT has data/proteus_windows.npz and runs/shuffle_perms.npz; ORDER is full0, within0, ...; one window per call, per-stream
state (person|session). Output: row-aligned posteriors (2655, 3) and the delivery. For full0 also compares against the
Lane C2 bank (ensemble_C_held_new.npz full0_members) and fails loudly above --atol."""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
exec((HERE / "cb_member.py.inc").read_text())  # same source as the package fragment
from cb_export import export_blob  # noqa: E402


HALF = False


def run(root, ckpts, order, seed, people, bank=None, atol=1e-3):
    with np.load(root / "data/proteus_windows.npz", allow_pickle=False) as z:
        x, subject, session = z["X"], np.array([s.rsplit("/", 1)[-1] for s in z["subject"]]), z["session"]
    with np.load(root / "runs/shuffle_perms.npz", allow_pickle=False) as p:
        delivery = p[f"{order[:-1]}|{order[-1]}|all_loso"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = np.full((len(x), 3), np.nan)
    for person in people:
        ckpt = torch.load(ckpts / f"reve_full_C_{seed}_{person}.pt", map_location="cpu", weights_only=True)
        blob = io.BytesIO()
        torch.save(export_blob(ckpt, half=HALF), blob)
        member = CBMember([blob.getvalue()], dev)
        state = {}
        for row in delivery[subject[delivery] == person]:
            key = session[row]
            member.cov_sum, member.n = state.get(key, (np.zeros((41, 41)), 0))
            out[row] = member.prob(x[row])
            state[key] = (member.cov_sum, member.n)
        print("done", person, seed, order, flush=True)
    return out, delivery


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("root", type=Path); a.add_argument("ckpts", type=Path); a.add_argument("out", type=Path)
    a.add_argument("order"); a.add_argument("seed", type=int)
    a.add_argument("--people", default=",".join(f"P{i:02}" for i in range(1, 11)))
    a.add_argument("--bank", type=Path, help="Lane C2 ensemble_C_held_new.npz for the parity check (full orders)")
    a.add_argument("--atol", type=float, default=1e-3)
    a.add_argument("--half", action="store_true", help="ship fp16 weights (size test)")
    a = a.parse_args()
    HALF = a.half
    people = a.people.split(",")
    post, delivery = run(a.root, a.ckpts, a.order, a.seed, people)
    if a.bank:
        with np.load(a.bank, allow_pickle=False) as z:
            ref = z[f"{a.order}_members"][a.seed]
            assert (z["index"] == np.arange(2655)).all() and (z[f"{a.order}_delivery"] == delivery).all()
        with np.load(a.root / "data/proteus_windows.npz", allow_pickle=False) as z:
            mask = np.isin([s.rsplit("/", 1)[-1] for s in z["subject"]], people)
        diff = np.abs(post[mask] - ref[mask])
        print("PARITY max|diff| %.3e mean %.3e over %d rows" % (diff.max(), diff.mean(), mask.sum()), flush=True)
        assert diff.max() <= a.atol, f"member differs from bank by {diff.max()}"
    np.savez(a.out, post=post, delivery=delivery, order=a.order, seed=a.seed)
