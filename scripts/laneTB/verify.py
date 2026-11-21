"""Lane TB verify: re-score Team B blends on fresh orders over the hardcb_safe or the hard-only fold base.

python verify.py --base hardcb|hard --fold new_P01 --orders 71124,71125 --out DIR [--smoke]
Writes DIR/<base>/<fold>/<arm>/pred_<mode>_<held>_<first order>.npz (rows_/labels_ keys, source and weights receipts)
and DIR/<base>/audit_<fold>.json (Team B training roster recomputed from the data). --smoke: the base arm must reproduce
a stored tape on a pinned order and nine25 must change labels on that stream; nothing is scored.
"""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/laneTB")]
from run import D, DEVICE, stream_session, timed_load  # noqa: E402  (adds harness and team_b to sys.path)
import build as B  # noqa: E402
from check_r1a import digest, fold_fixture  # noqa: E402

W = Path("/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs")
FOLDS = W / "production-20261120-32a94ab7/hardcb"
WEIGHTS = W / "laneTB-21nov/tb"
HARD_ZIP = W / "final-r1a-21nov/zips/final_stack_hard_safe_st.zip"
HARD_SHA = "239c86bd18d0a7e9d0216dda1fb95d9b918bf6e933c0383322446aed37a83c1e"
PARENT = {"hardcb": "StackEnsemble", "hard": "HardWaveEnsemble"}
STORED = {"hardcb": (W / "laneTB-21nov/streams/new_P01/base/pred_new_P01.npz", 71120),
          "hard": (W / "overnight-21nov-r1abases/out/hard/pred_new_P01_71000_71007.npz", 71000)}
ARMS = ("base", "nine15", "nine25")


def revised(original, weight, members, blob, parent):
    if parent == "StackEnsemble":
        return B.revised(original, weight, members, blob)  # the exact Lane TB / package code path
    original = B.safe_source(original)
    factory = f"model = {parent}(z,"
    if original.count(factory) != 1 or original.count(B.SOLVER) != 1:
        raise ValueError(f"expected one {parent} Solver factory")
    stack = (B.HERE / "stack.py.inc").read_text()
    if stack.count("(StackEnsemble)") != 1:
        raise ValueError("unexpected Team B subclass header")
    stack = stack.replace("(StackEnsemble)", f"({parent})").replace("__BLEND__", repr(weight))
    stack = stack.replace("__MEMBERS__", repr(members)).replace("__SHA__", repr(B.sha(blob)))
    encoder = B.encoder_source()
    inserted = f"_TEAM_B = {{}}\nexec(compile({encoder!r}, '<Team B encoder>', 'exec'), _TEAM_B)\n\n"
    src = original.replace(B.SOLVER, inserted + stack + "\n\n" + B.SOLVER)
    src = src.replace(factory, "model = TeamBlendStack(z,")
    compile(src, "submission.py", "exec")
    return src


def stage(base, fold, weights, work):
    """Arm packages: the fold's own assets, one changed submission.py, team_b.pt; base arm = blend 0."""
    src = FOLDS / fold / "pkg"
    if base == "hard":
        if digest(HARD_ZIP) != HARD_SHA:
            raise ValueError("hard ZIP digest differs")
        unpacked, fixture = work / "hardzip", work / "fixture"
        fixture.mkdir()
        with ZipFile(HARD_ZIP) as z:
            z.extract("submission.py", unpacked)
        fold_fixture(fixture, unpacked / "submission.py", src, "hard")
        src = fixture
    blobs = {3: B.bank(weights, 3)}
    pkgs = {}
    for arm in ARMS:
        weight, members = B.ARMS[arm]
        blob = blobs[members] if weight else b""
        pkg = work / arm
        pkg.mkdir()
        text = revised((src / "submission.py").read_text(), weight, members, blob, PARENT[base])
        (pkg / "submission.py").write_text(text)
        for path in src.iterdir():
            if path.name != "submission.py" and path.is_file():
                (pkg / path.name).symlink_to(path.resolve())
        (pkg / "team_b.pt").write_bytes(blob)
        pkgs[arm] = pkg
    return pkgs


def audit(data, fold, weights):
    mode, held = fold.split("_")
    pack = torch.load(weights, map_location="cpu", weights_only=True)
    people = D.PEOPLE if mode == "ret" else [p for p in D.PEOPLE if p != held]
    if pack["people"] != people or pack["first_only"] != ([held] if mode == "ret" else []) or pack.get("ext"):
        raise ValueError(f"Team B roster differs for {fold}: {pack['people']} {pack['first_only']}")
    rows = D.train_rows(data, pack["people"], pack["first_only"])
    streams = sorted(set(data["stream"][rows]))
    held_streams = [s for s in streams if s.startswith(held + "|")]
    if held_streams != (([f"{held}|1"]) if mode == "ret" else []):
        raise ValueError(f"held-out data in Team B training for {fold}: {held_streams}")
    return dict(fold=fold, people=pack["people"], first_only=pack["first_only"], ext=pack.get("ext"),
                train_windows=int(len(rows)), train_streams=streams, weights_sha=B.sha(weights.read_bytes()))


def check_class(model, base):
    names = [c.__name__ for c in type(model).__mro__[:2]]
    if names != ["TeamBlendStack", PARENT[base]] or type(model.ft).__name__ != "FineTune":
        raise ValueError(f"unexpected Solver class {names}")


def smoke(args, data, signals, pkgs):
    path, order = STORED[args.base]
    rows = D.plans(data, order, ["P01|1"])["P01|1"]
    with np.load(path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored[f"rows_{order}_1"], rows)
        expected = stored[f"labels_{order}_1"]
    labels = {}
    for arm in ("base", "nine25"):
        model, _ = timed_load(pkgs[arm], data, DEVICE)
        check_class(model, args.base)
        assert (model.team_b is None) == (arm == "base")
        labels[arm], _, meta = stream_session(model, signals, rows, DEVICE)
        print(arm, meta, flush=True)
    np.testing.assert_array_equal(labels["base"], expected)
    changed = int(np.count_nonzero(labels["base"] != labels["nine25"]))
    if not changed:
        raise ValueError("nine25 changed no label on the smoke stream")
    print(f"PASS {args.base}: base equals stored tape on {len(rows)} windows; nine25 changed {changed}", flush=True)


def scored(args, data, signals, pkgs, weights, source_sha):
    mode, held = args.fold.split("_")
    sessions = ["2", "3"] if mode == "ret" else sorted(set(data["session"][data["person"] == held]))
    orders = [int(s) for s in args.orders.split(",")]
    for arm, pkg in pkgs.items():
        model, _ = timed_load(pkg, data, DEVICE)
        check_class(model, args.base)
        res = {"source_sha": source_sha, "weights_sha": B.sha(weights.read_bytes())}
        for order in orders:
            for stream, rows in D.plans(data, order, [f"{held}|{s}" for s in sessions]).items():
                labels, _, meta = stream_session(model, signals, rows, DEVICE)
                suffix = f"{order}_{stream.split('|')[1]}"
                res.update({f"rows_{suffix}": rows, f"labels_{suffix}": labels})
                print(args.base, arm, args.fold, suffix, len(rows), meta, flush=True)
        out = args.out / args.base / args.fold / arm
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"pred_{mode}_{held}_{orders[0]}.npz", **res)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", choices=tuple(PARENT), required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--orders", default="")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    data = D.load(args.data)
    signals = torch.from_numpy(data["X"])
    weights = WEIGHTS / args.fold / "weights.pt"
    info = audit(data, args.fold, weights)
    (args.out / args.base).mkdir(parents=True, exist_ok=True)
    (args.out / args.base / f"audit_{args.fold}.json").write_text(json.dumps(info, indent=1) + "\n")
    with TemporaryDirectory() as tmp:
        pkgs = stage(args.base, args.fold, weights, Path(tmp))
        source_sha = B.sha((pkgs["base"] / "submission.py").read_bytes())
        if args.smoke:
            smoke(args, data, signals, pkgs)
        else:
            scored(args, data, signals, pkgs, weights, source_sha)


if __name__ == "__main__":
    main()
