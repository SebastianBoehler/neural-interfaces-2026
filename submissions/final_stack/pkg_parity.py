"""Package-level parity: stream one held person through the package Solver and compare CBMember posteriors with the bank.

usage: pkg_parity.py ZIP ROOT BANK.npz PERSON [--device cpu|cuda]   (ZIP built with that person's LOPO blobs, seeds 0-2)
Streams the person's sessions in the pinned full0 order, one window per predict() call on a fresh model copy per session;
records every CBMember.prob output and the v6 error counter; fails above --atol (bank = Lane C2 ensemble_C_held_new)."""
import argparse
import importlib.util
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
from benchmark_utils.streaming import fresh_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zip", type=Path); p.add_argument("root", type=Path); p.add_argument("bank", type=Path)
    p.add_argument("person"); p.add_argument("--device", default="cpu"); p.add_argument("--atol", type=float, default=1e-3)
    a = p.parse_args()
    with np.load(a.root / "data/proteus_windows.npz", allow_pickle=False) as z:
        x = torch.from_numpy(z["X"]); channels = z["ch_names"].tolist(); session = z["session"]
        subject = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
    with np.load(a.root / "runs/shuffle_perms.npz", allow_pickle=False) as z:
        delivery = z["full|0|all_loso"]
    with np.load(a.bank, allow_pickle=False) as z:
        ref = z["full0"]
        assert (z["full0_delivery"] == delivery).all() and (z["index"] == np.arange(len(ref))).all()
    with TemporaryDirectory() as tmp:
        zipfile.ZipFile(a.zip).extractall(tmp)
        spec = importlib.util.spec_from_file_location("pkg_submission", Path(tmp) / "submission.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        tap = []
        original = module.CBMember.prob
        module.CBMember.prob = lambda self, raw: tap.append(original(self, raw)) or tap[-1]
        template = module.Solver().load_model({"submission_dir": Path(tmp), "device": a.device, "ch_names": channels})
        rows = delivery[subject[delivery] == a.person]
        got, labels = {}, {}
        for key in dict.fromkeys(session[rows]):
            model = fresh_model(template)
            for r in rows[session[rows] == key]:
                n = len(tap)
                labels[int(r)] = int(model.predict(x[int(r):int(r) + 1].to(a.device))[0])
                assert len(tap) == n + 1, f"CBMember.prob called {len(tap) - n} times for window {r}"
                got[int(r)] = tap[-1]
            log = getattr(model, "log6", {})
            assert not log.get("err", 0), f"v6 errors {log}"
    index = np.array(sorted(got))
    diff = np.abs(np.stack([got[i] for i in index]) - ref[index])
    print(f"PARITY package vs bank: max|diff| {diff.max():.3e} mean {diff.mean():.3e} rows {len(index)} labels {np.bincount(list(labels.values()))}")
    assert diff.max() <= a.atol
    print("PASS")


if __name__ == "__main__":
    main()
