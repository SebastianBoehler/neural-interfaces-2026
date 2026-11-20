"""Injected faults must fail load_model loudly: usage selftest_inject.py PKG.zip MODE (hardcb | hard), run from the repo root."""
import importlib.util
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "2026-competition"), str(ROOT / "native/tools")]


def load(z, patch=None):
    with TemporaryDirectory() as s:
        ZipFile(z).extractall(s)
        spec = importlib.util.spec_from_file_location("sub", Path(s) / "submission.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if patch:
            patch(mod)
        return mod.Solver().load_model({"submission_dir": Path(s), "device": "cpu"})


def drift(fn):  # wrong but finite, normalised posteriors: only the reference comparison can catch this
    def wrapped(self, x):
        p = fn(self, x)
        q = np.roll(p, 1, axis=-1)
        return q
    return wrapped


def raiser(self, x):
    raise RuntimeError("injected")


def main(z, mode):
    cases = [("hard raises", lambda m: setattr(m.HardWaveEnsemble, "hard_probability", raiser)),
             ("hard drift", lambda m: setattr(m.HardWaveEnsemble, "hard_probability", drift(m.HardWaveEnsemble.hard_probability)))]
    if mode == "hardcb":
        cases += [("CBMember.prob raises", lambda m: setattr(m.CBMember, "prob", raiser)),
                  ("CBMember.prob drift", lambda m: setattr(m.CBMember, "prob", drift(m.CBMember.prob)))]
    load(z)
    print("clean load OK")
    ok = True
    for name, patch in cases:
        try:
            load(z, patch)
            print(f"{name}: LOADED (self-test missed it)")
            ok = False
        except Exception as e:
            print(f"{name}: load_model failed as required: {str(e)[:110]}")
    print("INJECTION OK" if ok else "INJECTION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
