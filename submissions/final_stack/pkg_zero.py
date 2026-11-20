"""Zero-change check: the stack package with CBraMod at weight 0 and the hard actors switched off (works for the cb and the hardcb zip) must reproduce stock cap250 exactly.

usage: pkg_zero.py STACK_ZIP(cb variant) ROOT PERSON...   Streams the person's visit-1 and visit-2 sessions (first 200 windows
of each in recording order, fresh copy per session) through stock and stack; compares the v6 step posteriors and labels."""
import importlib.util
import os
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
from benchmark_utils.streaming import fresh_model  # noqa: E402

WINDOWS = int(os.environ.get("PKG_ZERO_WINDOWS", 200))
STOCK = ROOT / "runs/sealed/final_zips/v66_cp1_cap250.zip"


def stream(zip_path, mutate, x, channels, sessions):
    with TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp)
        spec = importlib.util.spec_from_file_location("zero_submission", Path(tmp) / "submission.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        mutate(module)
        tap, step = [], module.Ensemble6.step
        module.Ensemble6.step = lambda self, w: tap.append(step(self, w)) or tap[-1]
        template = module.Solver().load_model({"submission_dir": Path(tmp), "device": "cpu", "ch_names": channels})
        labels = []
        for rows in sessions:
            model = fresh_model(template)
            labels += [int(model.predict(x[i:i + 1])[0]) for i in rows]
        return np.stack(tap), np.array(labels)


def main(zip_path, root, people):
    with np.load(Path(root) / "data/proteus_windows.npz", allow_pickle=False) as z:
        x = torch.from_numpy(z["X"]); channels = z["ch_names"].tolist(); session = z["session"]
        subject = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
    sessions = [np.flatnonzero((subject == p) & (session == s))[:WINDOWS] for p in people for s in np.unique(session[subject == p])[:2]]
    def zero_out(m):  # CBraMod weight 0 and no hard actors: the raw slot is the five original members again
        m.StackEnsemble.CB_MIX = 0.0
        m.StackEnsemble.HARD_FILES = ()

    ref_p, ref_l = stream(STOCK, lambda m: None, x, channels, sessions)
    new_p, new_l = stream(zip_path, zero_out, x, channels, sessions)
    diff = float(np.abs(ref_p - new_p).max())
    print(f"ZERO-CHANGE stock vs stack(CB_MIX=0): windows {len(ref_l)} max|step posterior diff| {diff:.3e} labels equal {bool((ref_l == new_l).all())}")
    assert (ref_l == new_l).all() and diff < 1e-9, "stack at zero weight is not stock"
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
