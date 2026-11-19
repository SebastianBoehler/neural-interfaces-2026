"""Build the final-stack package from exact cap250: hard pair (pinned hard trial ZIP) and/or CBraMod-C member blobs.

usage: build_final.py VARIANT OUT.zip CB_DIR   VARIANT: hardcb | cb | (hard = the existing F-C25p trial ZIP, not rebuilt)
CB_DIR holds cb_s0.pt..cb_s2.pt (export_blob dicts); their SHA-256 are written into the submission source."""
import argparse
import hashlib
import io
import os
import sys
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARD_DIR = Path(os.environ.get("STACK_HARD_DIR", ROOT / "friday/cap250_hard_package"))  # build_hard.py + hard_model.py.inc
sys.path.insert(0, str(HARD_DIR))
from build_hard import CODE, FACTORY, MARKER, STOCK  # noqa: E402

STOCK_ZIP = ROOT / "runs/sealed/final_zips/v66_cp1_cap250.zip"
HARD_ZIP = ROOT / "runs/experimental/c25p/v66_cap250_hard_trial.zip"
HARD_ZIP_SHA = "6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc"
HARD_ASSETS = ("hard_s1c_s0.pt", "hard_s1c_s1.pt")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def revised(original, variant, cb_files, hard_files=None):
    if sha(original) != CODE:
        raise ValueError("cap250 source bytes differ")
    src = original.decode("utf-8")
    if src.count(MARKER) != 1 or src.count(FACTORY) != 1:
        raise ValueError("expected one official Solver factory")
    hard = (HARD_DIR / "hard_model.py.inc").read_text().rstrip()
    cb = (HERE / "cb_member.py.inc").read_text().rstrip()
    stack = (HERE / "stack_model.py.inc").read_text().rstrip().replace("__CB_FILES__", repr(tuple(cb_files)))
    if variant == "cb":  # no hard actors: the (unused) hard class keeps no pinned files
        hard_files = ()
    if hard_files is not None:  # per-fold hard actors (returning instrument) or none
        stack = stack.replace("    CB_MIX = 0.2", f"    CB_MIX = 0.2\n    HARD_FILES = {tuple(hard_files)!r}")
    src = src.replace(MARKER, f"{hard}\n\n\n{cb}\n\n\n{stack}\n\n\n{MARKER}")
    src = src.replace(FACTORY, FACTORY.replace("Ensemble6", "StackEnsemble"))
    compile(src, "submission.py", "exec")
    return src.encode("utf-8")


def assemble(variant, out, cb_dir):
    assert variant in ("hardcb", "cb")
    stock = STOCK_ZIP.read_bytes()
    if sha(stock) != STOCK or sha(HARD_ZIP.read_bytes()) != HARD_ZIP_SHA:
        raise ValueError("unexpected stock or hard trial ZIP")
    if out.exists():
        raise FileExistsError(out)
    cb_assets = {f"cb_s{s}.pt": (Path(cb_dir) / f"cb_s{s}.pt").read_bytes() for s in range(3)}
    cb_files = [(n, sha(b)) for n, b in cb_assets.items()]
    assets = dict(cb_assets)
    if variant == "hardcb":
        with ZipFile(HARD_ZIP) as hz:
            assets = {n: hz.read(n) for n in HARD_ASSETS} | assets
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as scratch, ZipFile(STOCK_ZIP) as original:
        names = original.namelist()
        staged = Path(scratch) / "pkg.zip"
        with ZipFile(staged, "x") as pending:
            for entry in original.infolist():
                b = original.read(entry.filename)
                pending.writestr(copy(entry), revised(b, variant, cb_files) if entry.filename == "submission.py" else b)
            for name, b in assets.items():
                e = ZipInfo(name)
                e.compress_type = ZIP_DEFLATED
                e.external_attr = 0o100644 << 16
                pending.writestr(e, b)
        with ZipFile(staged) as pkg:
            assert pkg.namelist() == names + list(assets) and pkg.testzip() is None
            for n in names:
                assert n == "submission.py" or original.read(n) == pkg.read(n), n
        os.link(staged, out)
    return sha(out.read_bytes())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant"); p.add_argument("out", type=Path); p.add_argument("cb_dir", type=Path)
    a = p.parse_args()
    print(assemble(a.variant, a.out, a.cb_dir))
