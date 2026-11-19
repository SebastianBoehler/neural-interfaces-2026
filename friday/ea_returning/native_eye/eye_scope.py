"""Pinned, label-free source and decision scopes for native Eye qualification."""

import hashlib
from pathlib import Path

import numpy as np


BASE = Path("/mnt/lustre/work/gehler/gfh527")
REPO = BASE / "neural-interfaces-2026"
DATA = REPO / "data/proteus_windows.npz"
OCUNET = REPO / "studies/12nov/ocunet.py"
PRODUCER = REPO / "scripts/train_v6_members.py"
FIT = BASE / "ni26_friday_return_eye_fit_v1_20261119"
REPLAY = BASE / "ni26_friday_return_eye_native_v1_20261119"
INNER = BASE / "ni26_friday_returning_native_inner_v2_20261119"
BUNDLE = BASE / "ni26_js321_20261117_returning_bundle2"
ORDER_FILE = REPO / "runs/shuffle_perms.npz"
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
PEOPLE = ("P04", "P10")
VISITS = (("P04", "2", 189), ("P04", "3", 188), ("P10", "2", 186), ("P10", "3", 199))
FOLDS = {
    "P04": ("view_fit1", 2278, 377, 189,
            "8f52ab55d4dea7eca9b43b610fad2c645c1f3029372570d05ab71b2187123675",
            "ca4a869ff981142b692b2a4e2ecb44f075ea2494e398280f103c9a9a36bcea2b",
            "07db73505fa397db8afdcff984bc6d6a408a04c569681d0528946987c5be6d1c"),
    "P10": ("view_fit_p10_1", 2270, 385, 192,
            "ab59b6e9b2bd1104db9bfec80acf4e075a3151819f9b1d33c15d26f14e46a833",
            "db255fd45af6d17fa4ba1829781bac5a92a08afa1bec4e5bf85266b56ea84790",
            "8aaa3c2c7d59a3851fb0c6cb0ece879771484a9a0151885c645df7b889e439b2"),
}
ORIGINAL_TRACES = (
    "567afb369be1d090609be95c6abf5b139f78fcb4f0016e854923e2182d799478",
    "1540c7850e6c659e073a0ec28849629cc43ff4ac980f6ee19209e11376bdf9b6",
    "6ea7b6f830c57085f612f27d4a7e2b62893df2cb2a8e58870ebc69a21bef320d",
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pin(path, digest):
    if sha(path) != digest:
        raise ValueError(f"byte identity differs for {path}")


def rows_sha(rows):
    return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()


def source_scope(who):
    name, n_source, n_held, n_own, view_sha, fit_sha, held_sha = FOLDS[who]
    pin(DATA, DATA_SHA)
    view = BASE / ("ni26_js321_20261117_returning_" + name) / f"source_view_{who}_s23.npz"
    pin(view, view_sha)
    with np.load(DATA, allow_pickle=False) as z:
        people, visits = z["subject"], z["session"]
        own = people == f"Dreyer2026Proteus/{who}"
        at_source = (visits == "1") | ~own
    source, held = np.flatnonzero(at_source), np.flatnonzero(~at_source)
    if ((len(source), len(held), int((at_source & own).sum())) != (n_source, n_held, n_own)
            or rows_sha(source) != fit_sha or rows_sha(held) != held_sha):
        raise ValueError(f"source exclusions differ for the real {who} fold")
    with np.load(view, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["idx_tr"], source)
    return source, held, people, visits
