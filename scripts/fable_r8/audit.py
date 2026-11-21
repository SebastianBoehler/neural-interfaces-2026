"""Run the signed stacks on clean fold assets, saving blind predictions and the CountFilter latch."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from protocol import CHUNKS, DATA_SHA, FOLDS
from capture import V, fixture_assets, normalized
from check_r1a import fold_fixture
from dose import sessions
from grow_replay import check, load
from patch_ba_six_fp16 import patch as fp16
from returning_diet import patch as quarter_patch
from returning_diet_worker import diet, modified, fresh, stage
from returning_em import patch as half_patch
from tape import replay, tap


def check_hss(model, n):
    if type(model).__name__ != "HardWaveEnsemble" or model.cf.t != n or model.cp.t != n:
        raise ValueError("unexpected hss model or inactive CountFilter/CSP")
    if model.cf.dead or model.cp.dead or model.cf.log["err"] or model.cp.log["err"]:
        raise ValueError(f"hss tail failed: {model.cf.log}, {model.cp.log}")


def pin(path, key, rows, labels, *, strict):
    with np.load(path, allow_pickle=False) as tape:
        np.testing.assert_array_equal(rows, tape[f"rows_{key}"])
        changed = np.count_nonzero(labels != tape[f"labels_{key}"])
    if strict and changed:
        raise ValueError(f"{path}/{key}: {changed} changed pinned labels")
    return changed


def capture_ba(model, signals, rows):
    state = []

    def owned(template):
        m = fresh(template)
        state.append(m.cf)
        return m

    items = tap(model, signals, rows, owned, check, V.DEVICE)
    cf, = state
    if cf.t != len(rows) or cf.dead or cf.log["err"]:
        raise ValueError("native CountFilter detector did not consume the entire stream")
    base = np.asarray([i.label for i in items], dtype=np.int8)
    np.testing.assert_array_equal(base, replay(model, items, "w25")[0])
    recognised = np.asarray([i.recognised for i in items])
    if recognised.any():
        half = modified(model, items, "half")[0]
        quarter, scores, _ = diet(model, items, "quarter")
        np.testing.assert_array_equal(half[~recognised], base[~recognised])
        np.testing.assert_array_equal(quarter[~recognised], base[~recognised])
    else:
        half = quarter = base
        scores = [i.terminal for i in items]
    terminal = np.stack([normalized(p, False) for p in scores])
    np.testing.assert_array_equal(terminal.argmax(1), quarter)
    return base, half, quarter, terminal, -1 if cf.latch is None else cf.latch + 1, recognised


def main(a):
    if not torch.cuda.is_available() or V.digest(a.data) != DATA_SHA or V.digest(a.hss) != V.HARD_SHA:
        raise ValueError("CUDA and signed EEG / hss archive required")
    data = V.D.load(a.data)
    signals = torch.from_numpy(data["X"])
    folds = a.fixtures / "fixtures"
    chunk, index = divmod(a.task, len(FOLDS))
    fold = FOLDS[index]
    order_list = (71808, 72004) if a.smoke else CHUNKS[chunk]
    existing = folds / fold
    original = existing / "ba_six"
    receipts = json.loads((existing / "receipt.json").read_text())
    for arm, receipt in receipts.items():
        source = existing / arm
        if (receipt["data_sha"], receipt["fold"], receipt["source_sha"], receipt["bank_sha"]) != (
                DATA_SHA, fold, V.digest(source / "submission.py"), V.digest(source / "team_b.pt")):
            raise ValueError(f"changed fixture: {source}")
        fixture_assets(source, receipt)
    gpu = torch.cuda.get_device_name()
    turing = torch.cuda.get_device_capability() == (7, 5)
    p = dict(gpu=np.asarray(gpu), fold=np.asarray(fold))
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        with ZipFile(a.hss) as z:
            hss_source = Path(z.extract("submission.py", root))
        hss_folder = root / "hss"
        hss_folder.mkdir()
        fold_fixture(hss_folder, hss_source, existing / "nine25", "hard")
        hss, _ = V.timed_load(hss_folder, data, V.DEVICE)
        nine, _ = load(data, existing, "nine25", receipts["nine25"])
        fp16_source = fp16((original / "submission.py").read_text())
        stage(root / "ba_six", original, fp16_source)
        ba, _ = load(data, root, "ba_six", receipts["ba_six"])
        streams = sessions(data, fold)
        if a.smoke and a.hss_pin:
            rows = V.D.plans(data, 71124, streams[:1])[streams[0]]
            labels, _, _ = V.stream_session(hss, signals, rows, V.DEVICE, check=check_hss)
            key = f"71124_{streams[0].split('|')[1]}"
            pin(a.hss_pin / fold / "base" / f"pred_{fold}_71124.npz", key, rows, labels, strict=True)
        for order in order_list:
            for stream, rows in V.D.plans(data, order, streams[:1] if a.smoke else streams).items():
                key = f"{order}_{stream.split('|')[1]}"
                hard, _, _ = V.stream_session(hss, signals, rows, V.DEVICE, check=check_hss)
                nine_y, _, _ = V.stream_session(nine, signals, rows, V.DEVICE, check=check)
                ba_y, half, quarter, terminal, latch, rec = capture_ba(ba, signals, rows)
                if 71808 <= order <= 71811:
                    for arm, labels in (("nine25", nine_y), ("ba_six", ba_y)):
                        path = a.fixtures / "held" / fold / arm / f"pred_{fold}_{order - order % 2}.npz"
                        diffs = pin(path, key, rows, labels, strict=not turing)
                        print(f"PIN {fold}/{key}/{arm}: {diffs} changed labels ({gpu})", flush=True)
                if a.smoke:
                    smoke(a, data, signals, rows, hss, original, fp16_source, receipts["ba_six"],
                          hard, half, quarter, latch)
                for name, values in (("rows", rows), ("hss", hard), ("nine25_fc", nine_y),
                                     ("ba_six_fp16", ba_y), ("em_half", half), ("quarter", quarter),
                                     ("terminal", terminal), ("latch", latch), ("recognised", rec)):
                    p[f"{name}_{key}"] = np.asarray(values)
                print(f"PASS {fold}/{key}: {len(rows)} windows, count latch {latch}, recognised {rec.sum()}", flush=True)
    target = a.out / f"{fold}-{order_list[0]}.npz"
    if not a.smoke:
        with TemporaryDirectory(dir=a.out) as tmp:
            path = Path(tmp) / target.name
            np.savez_compressed(path, **p)
            path.rename(target)


def smoke(a, data, signals, rows, hss, original, source, receipt, hard, half, quarter, latch):
    from grow_fixture import prepare_zero

    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        prepare_zero(a.fixtures / "fixtures" / FOLDS[a.task] / "nine25", work / "zero")
        zero, _ = V.timed_load(work / "zero", data, V.DEVICE)
        same, _, _ = V.stream_session(zero, signals, rows[:65], V.DEVICE, check=check)
        np.testing.assert_array_equal(same, hard[:65])
        half_source = half_patch(source, 0.5)
        for arm, patched, expected in (("half", half_source, half), ("quarter", quarter_patch(half_source, 0.25), quarter)):
            stage(work / arm / "ba_six", original, patched)
            live, _ = load(data, work / arm, "ba_six", receipt)
            found = []

            def check_live(m, n):
                check(m, n)
                found.append(m.cf.latch)

            labels, _, _ = V.stream_session(live, signals, rows, V.DEVICE, check=check_live)
            np.testing.assert_array_equal(labels, expected)
            if arm == "half" and (found[0] + 1 if found[0] is not None else -1) != latch:
                raise ValueError("scheme latch differs from the native detector")
    prefix, _, _ = V.stream_session(hss, signals, rows[:65], V.DEVICE, check=check_hss)
    np.testing.assert_array_equal(prefix, hard[:65])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "hss", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("task", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--hss-pin", type=Path)
    main(p.parse_args())
