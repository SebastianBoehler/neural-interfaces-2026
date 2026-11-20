"""Prep for the overnight package comparison: hardlink stock fold packages, build hardcb_safe (guard patch on submission.py).
usage: pkgcmp_prep.py SRC_RUN DEST_RUN. Fails loudly unless every hardcb fold package holds exactly one guard text."""
import os
import sys
from pathlib import Path

OLD = ("            if self.hard_window is not None:\n"
       "                raise ValueError(\"the actual outer mixer did not consume the current waveform\")")
NEW = ("            if self.hard_window is not None and not self.dead:  # dead session: v5.1b fallback, keep running\n"
       "                raise ValueError(\"the actual outer mixer did not consume the current waveform\")")
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
folds = [d.name for d in sorted((src / "hardcb").iterdir()) if (d / "pkg").is_dir()]
assert len(folds) == 12, folds
for arm, srcarm in (("stock", "stock"), ("hardcb_safe", "hardcb")):
    for f in folds:
        out = dst / arm / f / "pkg"
        assert not out.exists(), out
        out.mkdir(parents=True)
        for p in (src / srcarm / f / "pkg").iterdir():
            os.link(p, out / p.name)
        if arm == "hardcb_safe":
            sub = out / "submission.py"
            text = sub.read_text()
            assert text.count(OLD) == 1, f
            sub.unlink()  # break the hardlink before writing, the production file stays untouched
            sub.write_text(text.replace(OLD, NEW))
            compile(sub.read_text(), str(sub), "exec")
print("prepared", len(folds) * 2, "packages")
