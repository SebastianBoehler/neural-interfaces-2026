"""Build a Lane T package from the final-stack hardcb package: the same code with FineTune replaced by TTTFineTune.

usage: build_ttt.py FORM OUT_ZIP [SRC_ZIP]   FORM light | heavy; SRC_ZIP defaults to runs/sealed/final_zips/final_stack_hardcb.zip
(unchanged, only submission.py is rewritten; every other member is copied byte for byte). Also: build_ttt.py FORM OUT_DIR SRC_DIR
for a package directory (returning fold packages)."""
import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo
from copy import copy

HERE = Path(__file__).resolve().parent
OLD = "FineTune(z, self.w, dev, root)"


def patch(src: bytes, form: str) -> bytes:
    s = src.decode("utf-8")
    assert form in ("light", "heavy") and s.count(OLD) == 1 and s.count("class CountFilter:") == 1
    inc = (HERE / "ttt_ft.py.inc").read_text().rstrip().replace('FORM = "light"  #', f'FORM = "{form}"  #')
    s = s.replace("class CountFilter:", inc + "\n\n\nclass CountFilter:", 1).replace(OLD, "TTTFineTune(z, self.w, dev, root)")
    compile(s, "submission.py", "exec")
    return s.encode("utf-8")


if __name__ == "__main__":
    form, out = sys.argv[1], Path(sys.argv[2])
    if out.suffix == ".zip":
        src = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE.parent / "runs/sealed/final_zips/final_stack_hardcb.zip"
        assert not out.exists()
        with ZipFile(src) as z, ZipFile(out, "x") as o:
            for e in z.infolist():
                b = z.read(e.filename)
                o.writestr(copy(e), patch(b, form) if e.filename == "submission.py" else b)
    else:
        src = Path(sys.argv[3])
        out.mkdir(parents=True)
        for f in src.iterdir():
            (out / f.name).write_bytes(patch(f.read_bytes(), form) if f.name == "submission.py" else f.read_bytes())
    print("built", out, hashlib.sha256(out.read_bytes()).hexdigest() if out.is_file() else "")
