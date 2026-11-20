"""Lane FT-labels prep: flipped copies of the fold packages (v6_ft_labels 0 -> 1), hardlinks except weights.npz (and submission.py
where patched). usage: ftl_prep.py SRC_PKG_RUN DEST_RUN [SMOKE_FOLD]   SRC_PKG_RUN holds {stock,hardcb_safe}/<fold>/pkg (overnight hardlink dirs).
Arms: stock = flag only; hardcb_safe = flag + GUARD patch (HardWaveEnsemble refuses v6r); hardcb_teach = flag + GUARD + TEACH (label chain reads
the hardcb mix). Also smoke_t0/new_P01: TEACH code with the ORIGINAL weights (flag 0), must be zero-change vs hardcb_safe.
Fails loudly unless every source weights.npz has v6_ft_labels == 0 and the copy differs in that one integer only."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftl_patches as P  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
folds = sorted(d.name for d in (src / "stock").iterdir() if (d / "pkg").is_dir())
assert len(folds) == 12, folds
ARMS = (("stock", "stock", [], True), ("hardcb_safe", "hardcb_safe", P.GUARD, True),
        ("hardcb_teach", "hardcb_safe", P.GUARD + P.TEACH, True), ("smoke_t0", "hardcb_safe", P.GUARD + P.TEACH, False))


def link(source, out, skip):
    out.mkdir(parents=True)
    for p in source.iterdir():
        if p.name not in skip:
            os.link(p, out / p.name)


for arm, base, patches, flip in ARMS:
    for f in (folds if arm != "smoke_t0" else ["new_P01"]):
        out = dst / arm / f / "pkg"
        assert not out.exists(), out
        source = src / base / f / "pkg"
        link(source, out, {"submission.py"} | ({"weights.npz"} if flip else set()))
        text = (source / "submission.py").read_text()
        (out / "submission.py").write_text(P.apply(text, patches) if patches else text)
        if not patches:
            os.unlink(out / "submission.py"); os.link(source / "submission.py", out / "submission.py")
        if not flip:
            continue
        with np.load(source / "weights.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        assert int(arrays["v6_ft_labels"]) == 0, (arm, f, arrays["v6_ft_labels"])
        flipped = dict(arrays)
        flipped["v6_ft_labels"] = np.asarray(1, dtype=arrays["v6_ft_labels"].dtype).reshape(arrays["v6_ft_labels"].shape)
        np.savez(out / "weights.npz", **flipped)
        with np.load(out / "weights.npz", allow_pickle=False) as z:
            assert sorted(z.files) == sorted(arrays), (arm, f)
            for k in arrays:
                if k == "v6_ft_labels":
                    assert int(z[k]) == 1 and z[k].dtype == arrays[k].dtype and z[k].shape == arrays[k].shape
                else:
                    assert z[k].dtype == arrays[k].dtype and z[k].tobytes() == arrays[k].tobytes(), (arm, f, k)
        print(arm, f, "flipped, other", len(arrays) - 1, "arrays identical, patches", len(patches), flush=True)
print("prepared flipped packages")
