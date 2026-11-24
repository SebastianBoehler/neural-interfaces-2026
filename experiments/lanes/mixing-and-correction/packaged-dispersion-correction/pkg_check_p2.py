"""Lane P2 package checks (CPU, real windows, official fresh_model per session, strictly the first WINDOWS windows of each session).

usage: pkg_check_p2.py TEST ZIP REF_ZIP ROOT PERSON...   (WINDOWS via PKG_P2_WINDOWS, default 120; sessions = visits 1-2 of each person)
  zero : ZIP (ds_l5 or ds_all) with DISP_ON False, FBL2_ON False and the stack's own switches untouched equals REF_ZIP (the lite5 package it was built from)
         bitwise on every step posterior and label, and the [disp] counters stay empty.
  disp : ZIP active (as shipped): per session, the package output equals laneP/chain/segcorr.correct on the tapped uncorrected posteriors and segments
         (labels = argmax of the uncorrected posterior) to 1e-12, skipped == 0, active > 0, changed > 0; REF_ZIP is not used (pass the same path).
  fbl2 : ZIP built from a stock-FT hardcb package (BUILD_P2_ANY_FT=1) with CB_MIX=0, HARD_FILES=(), DISP_ON False, FBL2_ON True equals REF_ZIP (the
         friday FBL2 trial ZIP of cap250) bitwise (CPU)."""
import importlib.util
import os
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "2026-competition/tracks/bci_decoding"), str(ROOT / "laneP2/chain")]
from benchmark_utils.streaming import fresh_model  # noqa: E402
import segcorr  # noqa: E402

WINDOWS = int(os.environ.get("PKG_P2_WINDOWS", 120))


def load(zip_path, tmp, channels):
    zipfile.ZipFile(zip_path).extractall(tmp)
    spec = importlib.util.spec_from_file_location(f"p2_{abs(hash((str(zip_path), tmp)))}", Path(tmp) / "submission.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def stream(zip_path, mutate, x, channels, sessions, tap_disp=False):
    with TemporaryDirectory() as tmp:
        module = load(zip_path, tmp, channels)
        mutate(module)
        tap, step = [], module.Ensemble6.step
        module.Ensemble6.step = lambda self, w: tap.append(step(self, w)) or tap[-1]
        rec = []
        if tap_disp:
            dstep = module.DispCorrection.step
            def wrapped(self, p, seg):
                out, active = dstep(self, p, seg); rec.append((p.copy(), seg.copy(), out.copy(), active)); return out, active
            module.DispCorrection.step = wrapped
        template = module.Solver().load_model({"submission_dir": Path(tmp), "device": "cpu", "ch_names": channels})
        labels, logs, bounds = [], [], []
        for rows in sessions:
            model = fresh_model(template)
            bounds.append(len(rec))
            labels += [int(model.predict(x[i:i + 1])[0]) for i in rows]
            logs.append(dict(getattr(model, "disp_log", {})))
        return np.stack(tap), np.array(labels), rec, logs, bounds


def main(test, zip_path, ref_path, root, people):
    with np.load(Path(root) / "data/proteus_windows.npz", allow_pickle=False) as z:
        x = torch.from_numpy(z["X"]); channels = z["ch_names"].tolist(); session = z["session"]
        subject = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
    sessions = [np.flatnonzero((subject == p) & (session == s))[:WINDOWS] for p in people for s in np.unique(session[subject == p])[:2]]
    if test == "zero":
        def off(m):
            m.DispEnsemble.DISP_ON = False; m.DispEnsemble.FBL2_ON = False
        ref_p, ref_l, *_ = stream(ref_path, lambda m: None, x, channels, sessions)
        new_p, new_l, _, logs, _ = stream(zip_path, off, x, channels, sessions)
        print(f"ZERO windows {len(ref_l)} max|diff| {np.abs(ref_p - new_p).max():.3e} bitwise {np.array_equal(ref_p, new_p)} labels equal {bool((ref_l == new_l).all())} disp logs {logs}")
        assert np.array_equal(ref_p, new_p) and (ref_l == new_l).all() and all(l.get("n", 0) == 0 for l in logs)
    elif test == "gate":  # dsu zip on recognised sessions: equals the ungated zip with DISP_ON False bitwise (REF_ZIP unused), [disp] gated > 0; UNREC_ONLY switched off equals the ungated ds zip (env P2_UNGATED) bitwise
        ref_p, ref_l, *_ = stream(os.environ["P2_UNGATED"], lambda m: setattr(m.DispEnsemble, "DISP_ON", False), x, channels, sessions)  # ungated zip, correction off (FBL2 as shipped)
        new_p, new_l, _, logs, _ = stream(zip_path, lambda m: None, x, channels, sessions)
        print(f"GATE windows {len(ref_l)} max|diff| {np.abs(ref_p - new_p).max():.3e} bitwise {np.array_equal(ref_p, new_p)} labels equal {bool((ref_l == new_l).all())} logs {logs}")
        assert np.array_equal(ref_p, new_p) and (ref_l == new_l).all() and all(l["gated"] > 0 for l in logs)
        off = lambda m: setattr(m.DispEnsemble, "UNREC_ONLY", False)
        a_p, a_l, *_ = stream(zip_path, off, x, channels, sessions)
        u_p, u_l, *_ = stream(os.environ["P2_UNGATED"], lambda m: None, x, channels, sessions)
        print(f"GATE-OFF equals ungated zip: bitwise {np.array_equal(a_p, u_p)} labels equal {bool((a_l == u_l).all())}")
        assert np.array_equal(a_p, u_p) and (a_l == u_l).all()
    elif test == "fbl2":
        def zero(m):
            m.StackEnsemble.CB_MIX = 0.0; m.StackEnsemble.HARD_FILES = (); m.DispEnsemble.DISP_ON = False; m.DispEnsemble.FBL2_ON = True
        ref_p, ref_l, *_ = stream(ref_path, lambda m: None, x, channels, sessions)
        new_p, new_l, *_ = stream(zip_path, zero, x, channels, sessions)
        d = np.abs(ref_p - new_p).max()
        print(f"FBL2 windows {len(ref_l)} max|diff| {d:.3e} bitwise {np.array_equal(ref_p, new_p)} labels equal {bool((ref_l == new_l).all())}")
        assert (ref_l == new_l).all() and d < 1e-9
    else:
        new_p, new_l, rec, logs, bounds = stream(zip_path, lambda m: None, x, channels, sessions, tap_disp=True)
        bounds.append(len(rec)); worst = 0.0; changed = 0; moved = 0.0
        for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            p = np.stack([r[0] for r in rec[a:b]]); seg = np.stack([r[1] for r in rec[a:b]]); out = np.stack([r[2] for r in rec[a:b]])
            ref, act = segcorr.correct(p, p.argmax(1), segcorr.features(seg, "disp"))
            assert np.array_equal(act, [r[3] for r in rec[a:b]])
            worst = max(worst, float(np.abs(ref - out).max())); moved = max(moved, float(np.abs(out - p).max())); changed += int((out.argmax(1) != p.argmax(1)).sum())
        print(f"DISP sessions {len(sessions)} windows {len(new_l)} streaming-vs-batch max|diff| {worst:.2e} logs {logs} changed {changed} max|corrected-uncorrected| {moved:.2e}")
        assert worst < 1e-12 and all(l["skipped"] == 0 and l["active"] > 0 and l["n"] == len(sessions[i]) for i, l in enumerate(logs)) and moved > 1e-6
    print("PASS", test)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
