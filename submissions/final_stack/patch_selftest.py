"""Derive a package whose constructor fails loudly when CBraMod or the hard actors do not reproduce their posteriors.

usage: patch_selftest.py MODE IN.zip IN_SHA256 OUT.zip     MODE: hardcb (StackEnsemble) | hard (HardWaveEnsemble)
Review of 20 Nov: if braindecode drifts or a member fails at predict time on the worker, every session goes dead and the
run ends as plain v5.1b (~0.82), which spends the last upload; a Failed run costs no quota. The inserted block runs two
fixed deterministic 41x480 windows through CBMember.prob (hardcb only) and hard_probability on the real device and
compares with posteriors computed here, on CPU, from the same package (atol 1e-3). Any error or mismatch raises in
load_model; on success the state is restored (cb.reset(), hard_actors = None), so labels are unchanged.
Only the submission.py text changes; every other entry is copied byte for byte. Run patch_dead.py first."""
import hashlib
import sys
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ATOL = 1e-3
ANCHOR_HARD = ("        if not np.allclose(self.hard_weights, 1 / 90, rtol=0, atol=1e-16):\n"
               "            raise ValueError(\"the original outer raw contribution has different weight\")\n")
ANCHOR_STACK = ("        if not self.cb_blobs and not self.HARD_FILES:\n"
                "            raise ValueError(\"the stack package has no member\")\n")
WINDOW = (  # integer-hash noise plus channel sines: no RNG stream, so identical under any numpy
    "        _i = np.arange(41 * 480, dtype=np.uint64).reshape(41, 480)\n"
    "        _u = ((_i * np.uint64(2654435761) + np.uint64(12345)) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32\n"
    "        _s = np.sin(2 * np.pi * (4.0 + 0.9 * np.arange(41))[:, None] * (np.arange(480) / 120.0)[None])\n"
    "        _xs = [(12.0 * (_u - 0.5) + a * _s).astype(np.float32) for a in (0.0, 2.0)]  # noise only, then weak sines\n")
HEAD = ("        # load-time self-test (patch_selftest.py): fixed windows, real device, CPU reference posteriors\n"
        "        # compared at atol {atol}; any error or mismatch fails load_model, state restored afterwards\n")
CB_TEST = ("        _ref_cb = np.array({cb})\n"
           "        try:\n"
           "            _got_cb = np.stack([self.cb.prob(_x) for _x in _xs])\n"
           "        except Exception as e:\n"
           "            raise RuntimeError(f\"load-time self-test: CBraMod member raised {{e!r}}\") from e\n"
           "        if not np.allclose(_got_cb, _ref_cb, rtol=0, atol={atol}):\n"
           "            raise RuntimeError(f\"load-time self-test: CBraMod posterior {{_got_cb}} != reference {{_ref_cb}}\")\n"
           "        self.cb.reset()\n")
HARD_TEST = ("        _ref_hard = np.array({hard})\n"
             "        try:\n"
             "            _got_hard = np.stack([self.hard_probability(_x) for _x in _xs])\n"
             "        except Exception as e:\n"
             "            raise RuntimeError(f\"load-time self-test: hard actors raised {{e!r}}\") from e\n"
             "        if not np.allclose(_got_hard, _ref_hard, rtol=0, atol={atol}):\n"
             "            raise RuntimeError(f\"load-time self-test: hard posterior {{_got_hard}} != reference {{_ref_hard}}\")\n"
             "        self.hard_actors = None\n")


def windows():
    ns = {"np": np}
    exec(WINDOW.replace("        ", "", 1).replace("\n        ", "\n"), ns)
    return ns["_xs"]


def reference(zin, mode):
    """Posteriors of the unpatched package on the fixed windows, CPU, through the package's own member code."""
    sys.path[:0] = [str(ROOT / "2026-competition"), str(ROOT / "native/tools")]
    from runtime_reference import load_solver
    xs = windows()
    with TemporaryDirectory() as scratch:
        ZipFile(zin).extractall(scratch)
        ens = load_solver(Path(scratch), "cpu")
        ref = {"hard": np.stack([ens.hard_probability(x) for x in xs]).tolist()}
        ens.hard_actors = None
        if mode == "hardcb":
            ref["cb"] = np.stack([ens.cb.prob(x) for x in xs]).tolist()
    for v in ref.values():
        assert np.isfinite(v).all() and np.allclose(np.sum(v, -1), 1.0, atol=1e-6), v
    return ref


def block(mode, ref):
    cb = CB_TEST.format(cb=repr(ref["cb"]), atol=ATOL) if mode == "hardcb" else ""
    return HEAD.format(atol=ATOL) + WINDOW + cb + HARD_TEST.format(hard=repr(ref["hard"]), atol=ATOL)


def main(mode, src, digest, out):
    if mode not in ("hardcb", "hard"):
        raise ValueError(mode)
    src, out = Path(src), Path(out)
    if hashlib.sha256(src.read_bytes()).hexdigest() != digest:
        raise ValueError("input ZIP SHA-256 differs")
    if out.exists():
        raise FileExistsError(out)
    anchor = ANCHOR_STACK if mode == "hardcb" else ANCHOR_HARD
    ref = reference(src, mode)
    print("reference posteriors", ref)
    with ZipFile(src) as zin, ZipFile(out, "x") as zout:
        for e in zin.infolist():
            b = zin.read(e.filename)
            if e.filename == "submission.py":
                text = b.decode("utf-8")
                if text.count(anchor) != 1 or "load-time self-test" in text:
                    raise ValueError("expected exactly one constructor end and no earlier self-test")
                if "the actual outer mixer did not consume the current waveform\")" in text and "and not self.dead" not in text:
                    raise ValueError("run patch_dead.py first")
                b = text.replace(anchor, anchor + block(mode, ref)).encode("utf-8")
                compile(b, "submission.py", "exec")
            zout.writestr(copy(e), b)
    with ZipFile(src) as a, ZipFile(out) as b:
        assert a.namelist() == b.namelist() and b.testzip() is None
        changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
        assert changed == ["submission.py"], changed
    print(hashlib.sha256(out.read_bytes()).hexdigest(), out)


if __name__ == "__main__":
    main(*sys.argv[1:])
