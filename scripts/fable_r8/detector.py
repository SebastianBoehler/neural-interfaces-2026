"""Measure the signed CountFilter latch using EEG views only, before opening accuracy."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import numpy as np

from protocol import DATA_SHA, FOLDS, ORDERS
from tapes import protocol


HSS_SHA = "239c86bd18d0a7e9d0216dda1fb95d9b918bf6e933c0383322446aed37a83c1e"


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def signed_hss(hss):
    if digest(hss) != HSS_SHA:
        raise ValueError("expected signed hss")
    with ZipFile(hss) as z:
        module = ModuleType("hss_detector")
        exec(compile(z.read("submission.py"), str(hss), "exec"), vars(module))
    return module


def fold_detector(module, fixtures, fold):
    package = fixtures / "fixtures" / fold
    receipt = json.loads((package / "receipt.json").read_text())["ba_six"]
    weights = package / "ba_six" / "weights.npz"
    if receipt["fold"] != fold or receipt["assets"][weights.name] != digest(weights):
        raise ValueError(f"changed detector assets for {fold}")
    with np.load(weights, allow_pickle=False) as z:
        return module.CountFilter(z)


def measure(data_path, fixtures, hss, folds):
    if digest(data_path) != DATA_SHA:
        raise ValueError("expected signed EEG")
    module = signed_hss(hss)
    data = protocol.load(data_path)
    windows = module.to_windows(data["X"], module.channel_map(data["ch_names"]))
    all_orders = {order: protocol.plans(data, order, np.unique(data["stream"]))
                  for orders in ORDERS.values() for order in orders}
    result = {}
    for fold in folds:
        cohort, person = fold.split("_")
        detector = fold_detector(module, fixtures, fold)
        streams = [s for s in np.unique(data["stream"]) if s.startswith(person + "|") and
                   (cohort == "new" or s.split("|")[1] in ("2", "3"))]
        views = {r: detector.view(windows[r]) for r in np.flatnonzero(np.isin(data["stream"], streams))}
        result[fold] = {}
        for order, plans in all_orders.items():
            for stream in streams:
                rows = plans[stream]
                detector.V = [views[r] for r in rows]
                first = next((n + 1 for n in detector.checks if n < len(rows) and detector.lag(n) < detector.thr[n]), -1)
                result[fold][f"{order}_{stream.split('|')[1]}"] = first
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "hss", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--fold", choices=FOLDS)
    a = p.parse_args()
    result = measure(a.data, a.fixtures, a.hss, (a.fold,) if a.fold else FOLDS)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote label-free detector windows for {len(result)} real folds")
