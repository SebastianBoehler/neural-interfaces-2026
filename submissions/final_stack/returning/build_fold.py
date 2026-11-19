"""Build the returning-instrument fold packages (visit 1 of WHO enrolled) with the stack members.

usage: build_fold.py WHO ARM OUT BASE_DIR HARD_DIR CB_DIR   ARM: hard | cb | hardcb
BASE_DIR: the Lane A fold base package (submission.py = cap250 source, sha pinned in build_final.CODE);
HARD_DIR: friday return_c20 actors (WHO/hard_s{0,1}.pt, whole person excluded); CB_DIR: cb_s0..2.pt from the
`returning` Lane C2 fits (P04/P10 visits 2-3 excluded)."""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_final import revised, sha  # noqa: E402


def build(who, arm, out, base, hard_dir, cb_dir):
    assert arm in ("hard", "cb", "hardcb") and not out.exists()
    out.mkdir(parents=True)
    for f in base.iterdir():
        if f.name != "submission.py":
            shutil.copyfile(f, out / f.name)  # dereferences the weights.npz symlink
    cb = {f"cb_s{s}.pt": (cb_dir / f"cb_s{s}.pt").read_bytes() for s in range(3)} if "cb" in arm else {}
    hard = {f"hard_s{s}.pt": (hard_dir / who / f"hard_s{s}.pt").read_bytes() for s in range(2)} if "hard" in arm else {}
    for name, b in {**cb, **hard}.items():
        (out / name).write_bytes(b)
    # "hard" alone: the stack class with no CB blobs (CB_FILES empty); "cb": no hard actors
    variant = "cb" if arm == "cb" else "hardcb"
    src = revised((base / "submission.py").read_bytes(), variant, [(n, sha(b)) for n, b in cb.items()],
                  [(n, sha(b)) for n, b in hard.items()])
    (out / "submission.py").write_bytes(src)


if __name__ == "__main__":
    who, arm, out, base, hard_dir, cb_dir = sys.argv[1:]
    build(who, arm, Path(out), Path(base), Path(hard_dir), Path(cb_dir))
    print("built", out)
