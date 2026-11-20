"""Derive a package whose dead-session fallback survives: HardWaveEnsemble.step no longer raises after a v6 error.

usage: patch_dead.py IN.zip IN_SHA256 OUT.zip
Bug (20 Nov review): after any v6-chain error step62 returns v5.1b's output without calling mix6, so the hard waveform
is never consumed and step() raised outside step62's try, ending the whole run instead of falling back. Only the
submission.py text of that guard changes; every other entry is copied byte for byte."""
import hashlib
import sys
from copy import copy
from pathlib import Path
from zipfile import ZipFile

OLD = ("            if self.hard_window is not None:\n"
       "                raise ValueError(\"the actual outer mixer did not consume the current waveform\")")
NEW = ("            if self.hard_window is not None and not self.dead:  # dead session: v5.1b fallback, keep running\n"
       "                raise ValueError(\"the actual outer mixer did not consume the current waveform\")")


def main(src, digest, out):
    src, out = Path(src), Path(out)
    if hashlib.sha256(src.read_bytes()).hexdigest() != digest:
        raise ValueError("input ZIP SHA-256 differs")
    if out.exists():
        raise FileExistsError(out)
    with ZipFile(src) as zin, ZipFile(out, "x") as zout:
        for e in zin.infolist():
            b = zin.read(e.filename)
            if e.filename == "submission.py":
                text = b.decode("utf-8")
                if text.count(OLD) != 1:
                    raise ValueError("expected exactly one hard-window guard")
                b = text.replace(OLD, NEW).encode("utf-8")
                compile(b, "submission.py", "exec")
            zout.writestr(copy(e), b)
    with ZipFile(src) as a, ZipFile(out) as b:
        assert a.namelist() == b.namelist() and b.testzip() is None
        changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
        assert changed == ["submission.py"], changed
    print(hashlib.sha256(out.read_bytes()).hexdigest(), out)


if __name__ == "__main__":
    main(*sys.argv[1:])
