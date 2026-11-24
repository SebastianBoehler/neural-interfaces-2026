"""Lane D3: drift-corrected variants of a v6 stack package (only submission.py is rewritten; every other file is copied byte for byte).
usage: build_drift.py ARM OUT_ZIP_OR_DIR SRC_ZIP_OR_DIR      ARMS: D0z (code in, mode 0: must equal the source bit for bit), D3f (fixed half-shift), D3g (gated)
The enrolment moments (d3_mu, d3_var, d3_n) must be in weights.npz of the package (niv/enrol.py d3=True, laneD3/add_d3.py); mode > 0 fails at load without them.
Every anchor must occur exactly once."""
import hashlib
import sys
from copy import copy
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
MODES = {"D0z": 0, "D3f": 1, "D3g": 2}


def once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:70])
    return s.replace(old, new)


def patch(src: bytes, arm: str) -> bytes:
    s = src.decode("utf-8")
    cls = (HERE / "drift.py").read_text().split("# --- pasted from here ---\n")[1]
    s = once(s, "class Ensemble6(Ensemble):", f"D3_MODE = {MODES[arm]}  # lane D3\n{cls}\n\nclass Ensemble6(Ensemble):")
    s = once(s, "        assert self.cp is None or self.cf is not None, \"v6_cp_on needs v6_cf_on\"\n",
             "        assert self.cp is None or self.cf is not None, \"v6_cp_on needs v6_cf_on\"\n        self.d3 = Drift(z, D3_MODE)  # lane D3\n")
    s = once(s, 'getattr(self, "cf", None), getattr(self, "cp", None)):', 'getattr(self, "cf", None), getattr(self, "cp", None), getattr(self, "d3", None)):')
    s = once(s, '                raise FloatingPointError("v6.2 output not finite")\n',
             '                raise FloatingPointError("v6.2 output not finite")\n'
             '            out = self.d3.apply(out, u, shadow, gc, gi)  # lane D3: after RouteB, bounded logit correction\n')
    compile(s, "submission.py", "exec")
    return s.encode("utf-8")


if __name__ == "__main__":
    arm, out, src = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    assert arm in MODES
    if src.is_file():
        assert out.suffix == ".zip" and not out.exists()
        with ZipFile(src) as z, ZipFile(out, "x") as o:
            for e in z.infolist():
                b = z.read(e.filename)
                o.writestr(copy(e), patch(b, arm) if e.filename == "submission.py" else b)
    else:
        out.mkdir(parents=True)
        for f in src.iterdir():
            (out / f.name).write_bytes(patch(f.read_bytes(), arm) if f.name == "submission.py" else f.read_bytes())
    print("built", arm, out, hashlib.sha256(out.read_bytes()).hexdigest() if out.is_file() else "")
