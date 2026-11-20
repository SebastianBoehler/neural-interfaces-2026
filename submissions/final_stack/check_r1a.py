"""Check the final R1a ZIPs through the real Solver, without opening task labels."""

import argparse
from contextlib import ExitStack
from hashlib import file_digest
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "laneR4"), str(ROOT / "cleanslate/harness"),
                str(ROOT / "2026-competition"), str(ROOT / "2026-competition/tracks/bci_decoding")]
from benchmark_utils.streaming import fresh_model
from build_keep import once, patch
from data import delivery
from patch_selftest import ANCHOR_HARD, ANCHOR_STACK
from stream import load_solver

MODES = ("hard", "hardcb")
PINNED = {
    "hard": ("239c86bd18d0a7e9d0216dda1fb95d9b918bf6e933c0383322446aed37a83c1e",
             "df31c7c9576fbbf8a0da3ff1959d77dc744bb0058052204f1401911c19b33322"),
    "hardcb": ("47faa27b25f6fc1fcbcc1c74f2be4b8b2aeb2a1e02c6c957d342eb6c91b9c2ec",
               "82cc75456c486529cbebe9b1b9ddb1621b8478a4ad69e81b6fa11a9a083c4861"),
}


def names(mode):
    parent = f"final_stack_{mode}_safe_st"
    return f"{parent}.zip", f"{parent}_r1a.zip"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify(zips, mode):
    archives = [zips / name for name in names(mode)]
    if tuple(map(digest, archives)) != PINNED[mode]:
        raise ValueError(f"{mode}: unexpected ZIP digest")
    with ZipFile(archives[0]) as parent, ZipFile(archives[1]) as new:
        if parent.namelist() != new.namelist() or new.testzip() is not None:
            raise ValueError(f"{mode}: bad CRC or changed member inventory")
        changed = [n for n in parent.namelist() if parent.read(n) != new.read(n)]
        if changed != ["submission.py"] or patch(parent.read(changed[0]), "R1a") != new.read(changed[0]):
            raise ValueError(f"{mode}: ZIP does not reproduce the Lane R4 builder")
    return archives


def fold_fixture(folder, production, fold, mode):
    """Apply the final executable to clean fold assets and their signed hashes.

    The production self-test references are specific to the production actors.
    The exact ZIP exercises those references in the smoke and returning checks.
    """
    pattern = r"(?ms)^    (?:HARD_FILES|CB_FILES) = (\(\n.*?^    \)|\([^\n]*\))$"
    for path in sorted(fold.iterdir()):
        if path.suffix == ".pt" and mode == "hard" and path.name.startswith("cb_s"):
            continue
        if path.name != "submission.py":
            (folder / path.name).symlink_to(path.resolve())
    files = sorted(folder.glob("hard_s*.pt"))
    signatures = {"HARD_FILES": tuple((p.name, digest(p)) for p in files)}
    if mode == "hardcb":
        signatures["CB_FILES"] = tuple((p.name, digest(p)) for p in sorted(folder.glob("cb_s*.pt")))
    text = production.read_text()
    for key, values in signatures.items():
        matches = [m[0] for m in re.finditer(pattern, text) if m[0].startswith(f"    {key} =")]
        if len(matches) != 1 or len(values) != (2 if key == "HARD_FILES" else 3):
            raise ValueError(f"unexpected fold actor inventory for {key}")
        text = once(text, matches[0], f"    {key} = {values!r}")
    anchor = ANCHOR_HARD if mode == "hard" else ANCHOR_STACK
    before, tail = text.split(anchor, 1)
    selftest, after = tail.split("\n    def reset_state", 1)
    if not selftest.startswith("        # load-time self-test") or not selftest.endswith("        self.hard_actors = None\n"):
        raise ValueError("unexpected production self-test block")
    (folder / "submission.py").write_text(before + anchor + "\n    def reset_state" + after)


def replay(template, windows, scores=None):
    start = perf_counter()
    model = fresh_model(template)
    if scores is not None:
        step = model.cp.step

        def trace(*args):
            posterior = step(*args)
            scores.append(np.array(posterior, copy=True))
            return posterior

        model.cp.step = trace
    labels, late = [], []
    for age, x in enumerate(windows):
        previous = model.ft.refits
        label = model.predict(x[None])
        if label.shape != (1,) or label.dtype != torch.int64 or label.item() not in (0, 1, 2):
            raise ValueError(f"invalid label at age {age}: {label}")
        labels.append(label.item())
        rec = model.log6["rec"]
        if rec >= 0 and age > rec and model.ft.refits != previous:
            late.append(age)
    if scores is not None:
        del model.cp.step
    if model.dead or model.log6["err"] or model.routeB.log["err"] or model.cp.nfail:
        raise RuntimeError("degraded Solver")
    return np.array(labels), rec, late, perf_counter() - start


def compare(models, windows, expected_late, trace_scores=False):
    scores = ([], []) if trace_scores else (None, None)
    before, after = [replay(model, windows, trace) for model, trace in zip(models, scores)]
    if before[1] != after[1] or before[2] or after[2] != expected_late:
        raise ValueError(f"unexpected recognition/adaptation: {before[1:3]} -> {after[1:3]}")
    changed = np.flatnonzero(before[0] != after[0])
    if len(changed) and (not expected_late or changed[0] < expected_late[0]):
        raise ValueError("labels changed before the R1a refit")
    info = dict(windows=len(windows), recognition=after[1], late_refits=after[2],
                changed=len(changed), baseline_s=before[3], r1a_s=after[3])
    if trace_scores:
        drift = np.max(np.abs(np.asarray(scores[0]) - np.asarray(scores[1])), axis=1)
        split = expected_late[0] if expected_late else len(windows)
        if drift.shape != (len(windows),) or not np.isfinite(drift).all() or drift[:split].max() > 2e-6:
            raise ValueError("terminal scores drift before R1a can act")
        if expected_late and drift[split:].max() <= 2e-6:
            raise ValueError("R1a refit had no effect on terminal scores")
        info["score_windows_changed"] = int(np.count_nonzero(drift > 2e-6))
        info["max_score_change"] = float(drift.max())
    return after[0], info


def load_templates(stack, zips, mode, channels, device, fold=None):
    models = []
    for name in names(mode):
        package = Path(stack.enter_context(TemporaryDirectory()))
        with ZipFile(zips / name) as archive:
            archive.extractall(package)
        if fold is not None:
            fixture = Path(stack.enter_context(TemporaryDirectory()))
            fold_fixture(fixture, package / "submission.py", fold, mode)
            package = fixture
        model = load_solver(package, channels, device)
        expected = "StackEnsemble" if mode == "hardcb" else "HardWaveEnsemble"
        if type(model).__name__ != expected or type(model.ft).__name__ != "FineTune":
            raise ValueError("unexpected Solver or fine-tuning class")
        models.append(model)
    return models


def run(a):
    verify(a.zips, a.mode)
    with np.load(a.data, allow_pickle=False) as data:
        sessions = np.char.add(np.char.add(data["subject"], "|"), data["session"])
        channels, x = data["ch_names"], torch.from_numpy(data["X"])
    seed = 71000
    order = delivery(sessions, seed)

    def rows(person, visit):
        return order[sessions[order] == f"Dreyer2026Proteus/{person}|{visit}"]

    with ExitStack() as stack:
        models = load_templates(stack, a.zips, a.mode, channels, a.device, a.fold)
        results = {}
        if a.fold is not None or a.smoke:
            selected = rows("P01", "1")
            if a.smoke:
                selected = selected[:8]
            _, info = compare(models, x[selected].to(a.device), [], a.scores)
            if info["recognition"] >= 0 or info["changed"] or len(selected) != (8 if a.smoke else 190):
                raise ValueError(f"held-new/smoke zero-change gate failed: {info}")
            results["P01/1"] = info
        else:
            # Warm both templates on the same recognised session before the timed pairs.
            compare(models, x[rows("P04", "2")].to(a.device), [100])
            for person in ("P04", "P10"):
                with np.load(a.tapes / f"pred_ret_{person}.npz", allow_pickle=False) as tape:
                    for visit in ("2", "3"):
                        key = f"{seed}_{visit}"
                        selected = rows(person, visit)
                        np.testing.assert_array_equal(tape[f"rows_{key}"], selected)
                        labels, info = compare(models, x[selected].to(a.device), [100], a.scores)
                        info["clean_tape_mismatches"] = int(np.count_nonzero(labels != tape[f"labels_{key}"]))
                        results[f"{person}/{visit}"] = info
            results["mean_delta_s"] = float(np.mean([r["r1a_s"] - r["baseline_s"] for r in results.values()]))
    a.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=MODES)
    p.add_argument("zips", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--device", choices=("cuda:0", "cpu"), required=True)
    p.add_argument("--fold", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--scores", action="store_true", help="Check the terminal posterior before argmax")
    p.add_argument("--tapes", type=Path)
    run(p.parse_args())
