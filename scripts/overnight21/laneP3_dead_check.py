"""Lane P3: dead-session guard check for a stack package. usage: laneP3_dead_check.py ROOT UNPATCHED.zip PATCHED.zip
ROOT holds 2026-competition/ and data/proteus_windows.npz. (1) guard text present/absent in each zip; (2) zero-change: labels and
Ensemble6.step posteriors of patched equal unpatched on P01 session 1 and P04 session 2 (no error injected); (3) one injected eye.step
error at window 4 of P01/1: unpatched must crash, patched must finish the session. CPU. Fails loudly (assert)."""
import importlib.util
import sys
import types
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(sys.argv[1]); sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
from benchmark_utils.streaming import fresh_model  # noqa: E402
N = int(__import__("os").environ.get("PKG_P3_WINDOWS", 190))
GUARD = "if self.hard_window is not None and not self.dead:"


def load(zp, tmp):
    zipfile.ZipFile(zp).extractall(tmp)
    spec = importlib.util.spec_from_file_location("p3_sub", Path(tmp) / "submission.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run(zp, x, ch, sessions, inject=None):
    with TemporaryDirectory() as tmp:
        m = load(zp, tmp)
        tap, step, s62 = [], m.Ensemble6.step, m.Ensemble6.step62
        m.Ensemble6.step = lambda self, w: tap.append(step(self, w)) or tap[-1]
        if inject is not None:
            def bad(self, w):
                if self.log6["n"] == inject and not self.dead:
                    self.eye = types.SimpleNamespace(step=lambda *_: (_ for _ in ()).throw(RuntimeError("injected eye error")))
                return s62(self, w)
            m.Ensemble6.step62 = bad
        template = m.Solver().load_model({"submission_dir": Path(tmp), "device": "cpu", "ch_names": ch})
        labels = []
        for rows in sessions:
            model = fresh_model(template)
            labels += [int(model.predict(x[i:i + 1])[0]) for i in rows]
        return np.stack(tap), np.array(labels)


def main(a, b):
    for name, zp in (("unpatched", a), ("patched", b)):
        t = zipfile.ZipFile(zp).read("submission.py").decode()
        print(f"guard present in {name}: {GUARD in t} (count {t.count(GUARD)})")
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as z:
        x = torch.from_numpy(z["X"]); ch = z["ch_names"].tolist(); ses = z["session"]
        sub = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
    rows = lambda p, s: np.flatnonzero((sub == p) & (ses == s))[:N]
    sessions = [rows("P01", "1"), rows("P04", "2")]
    pa, la = run(a, x, ch, sessions); pb, lb = run(b, x, ch, sessions)
    d = float(np.abs(pa - pb).max())
    print(f"ZERO-CHANGE windows {len(la)} labels equal {bool((la == lb).all())} max|step diff| {d:.3e}")
    assert (la == lb).all() and d == 0.0
    try:
        run(a, x, ch, [rows("P01", "1")[:12]], inject=4); print("unpatched with injected error: survived (UNEXPECTED)")
    except Exception as e:
        print("unpatched with injected error: crash:", repr(e)[:120])
    _, lc = run(b, x, ch, [rows("P01", "1")[:12]], inject=4)
    print("patched with injected error: finished", len(lc), "windows; labels before the error equal clean:", bool((lc[:4] == lb[:4]).all()))
    print("PASS")


main(sys.argv[2], sys.argv[3])
