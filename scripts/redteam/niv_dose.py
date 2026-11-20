"""Build and score the registered kappa40 NI-V B-ret confirmation."""

import argparse
import subprocess
import sys
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from report_ret import paired

SEEDS = range(71008, 71024)


def build(source, dest):
    with TemporaryDirectory(dir=dest.parent) as tmp, ZipFile(source) as original:
        stage = Path(tmp)
        base, patch = stage / "base", stage / "patch"
        original.extractall(base)
        with np.load(base / "weights.npz", allow_pickle=False) as weights:
            if float(weights["v6_tpri_kappa"]) != 20 or not int(weights["v6_tpri_on"]):
                raise ValueError(f"{source}: unexpected hardcbB prior")
        subprocess.run(
            [sys.executable, "-B", str(Path(__file__).with_name("patch_weights.py")),
             str(base), str(patch), "--set", "v6_tpri_kappa=40"], check=True)
        with ZipFile(stage / "candidate.zip", "x") as candidate:
            for entry in original.infolist():
                content = (patch / entry.filename).read_bytes() if entry.filename == "weights.npz" else original.read(entry)
                candidate.writestr(copy(entry), content)
        if dest.exists():
            raise FileExistsError(dest)
        (stage / "candidate.zip").rename(dest)


def compare(ref, arm, key):
    rows = arm[key + "|rows"]
    np.testing.assert_array_equal(rows, ref[key + "|rows"])
    ref_labels, arm_labels = ref[key + "|labels"], arm[key + "|labels"]
    before, after = ref[key + "|meta"], arm[key + "|meta"]
    if (rows.shape != arm_labels.shape or before[2] or after[2] or
            min(before[0], after[0]) < 0 or not np.isin(arm_labels, (0, 1, 2)).all()):
        raise ValueError(f"{key}: invalid or unrecognised NI-V stream")
    return rows, ref_labels, arm_labels


def smoke(reference, candidate):
    with np.load(reference, allow_pickle=False) as ref, np.load(candidate, allow_pickle=False) as arm:
        keys = [k.removesuffix("|rows") for k in arm.files if k.endswith("|rows")]
        if len(keys) != 1 or not keys[0].startswith("full|s71008|"):
            raise ValueError(f"unexpected smoke keys: {keys}")
        rows, before, after = compare(ref, arm, keys[0])
        print(f"NI-V smoke PASS: {keys[0]}, {len(rows)} paired windows, {(before != after).sum()} changed")


def score(data, reference, candidate):
    with np.load(data, allow_pickle=False) as windows:
        labels, subjects, visits = windows["y"], windows["subject"], windows["session"]
    names = sorted(set(subjects))
    if len(names) != 9:
        raise ValueError("expected nine Scherer people")
    for scheme in ("full", "within"):
        deltas = np.empty((len(names), len(SEEDS)))
        changed = total = 0
        for p, name in enumerate(names):
            expected = np.flatnonzero((subjects == name) & (visits == "2"))
            with np.load(reference / f"hardcbB_k{p}/ret.npz", allow_pickle=False) as ref, \
                 np.load(candidate / f"ret-{p}.npz", allow_pickle=False) as arm:
                for i, seed in enumerate(SEEDS):
                    rows, before, after = compare(ref, arm, f"{scheme}|s{seed}|{name}|2")
                    np.testing.assert_array_equal(np.sort(rows), expected)
                    correct_before, correct_after = before == labels[rows], after == labels[rows]
                    shifts = [correct_after[labels[rows] == j].mean() - correct_before[labels[rows] == j].mean()
                              for j in (0, 1, 2)]
                    deltas[p, i] = 100 * np.mean(shifts)
                    changed += np.count_nonzero(before != after)
                    total += len(rows)
        if not np.isfinite(deltas).all():
            raise ValueError(f"{scheme}: incomplete cells")
        _, interval = paired(deltas)
        print(f"{scheme}: {deltas.mean():+.2f} pp {interval}, {changed}/{total} labels changed; "
              f"per person {np.round(deltas.mean(axis=1), 2).tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "smoke", "score"))
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    {"build": build, "smoke": smoke, "score": score}[args.mode](*args.paths)
