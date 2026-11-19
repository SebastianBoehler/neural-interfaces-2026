"""Run frozen staged cap250 on Scherer and apply only the registered member adapter."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from data import permutation

from adapter import BoundedRaw, install


ORDERS = (411, 1411)
PACKAGE = Path("/package")


def stream(template, eeg, rows, arm):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    raw = install(model, PACKAGE)
    patches = {i: BoundedRaw(net, arm) for i, net in raw.items()} if arm != "stock" else {}
    for i, patch in patches.items():
        raw[i].proba = patch
    labels = []
    for row in rows:
        label = model.predict(eeg[int(row):int(row) + 1].to("cuda:0"))
        if label.device.type != "cpu" or label.dtype != torch.int64 or label.shape != (1,):
            raise ValueError("the actual worker did not emit one label")
        if model.dead or model.cp.dead or model.cf.dead:
            raise ValueError("worker entered a fallback")
        if not labels and patches and not all(patch.n == 1 for patch in patches.values()):
            raise ValueError("raw adapters did not all reach the first pre-mix call")
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    activity = replay.activity(model)
    if activity["errors"] or activity["windows"] != len(rows) or activity["ft_over"] or model.recB:
        raise ValueError(f"unexpected exact Scherer work or route: {activity}")
    receipts = {i: patch.receipt() for i, patch in patches.items()}
    if any(value["windows"] != len(rows) for value in receipts.values()):
        raise ValueError("an adapted member missed a delivered window")
    return dict(labels=labels, activity=activity, adapters=receipts, seconds=perf_counter() - start)


def main(index, arm, smoke):
    if torch.__version__ != "2.14.1+cu130" or "A100" not in torch.cuda.get_device_name():
        raise ValueError("requires the pinned A100 worker image")
    with np.load("/external.npz", allow_pickle=False) as z:
        eeg = torch.from_numpy(z["X"])
        people, days, channels = (z[name] for name in ("subject", "session", "ch_names"))
    if eeg.shape != (2130, 41, 480) or eeg.dtype != torch.float32:
        raise ValueError("unexpected Scherer EEG")
    who = sorted(set(people))[index]
    sessions = np.char.add(np.char.add(people, "|"), days)
    archive_path = Path(f"/prior/cap_{index}.json")
    archive = json.loads(archive_path.read_text())
    template = replay.load(PACKAGE, channels.tolist())
    output = {}
    for order in (ORDERS[:1] if smoke else ORDERS):
        delivery = permutation(sessions, None, "full", order)
        for day in sorted(set(days[people == who])):
            key = f"{who}|{day}|{order}"
            expected = archive[key]
            rows = delivery[(people == who) & (days == day)]
            if rows.tolist() != expected["rows"]:
                raise ValueError(f"matched causal delivery changed for {key}")
            result = stream(template, eeg, rows, arm)
            match = result["labels"] == expected["labels"]
            if result["activity"] != expected["activity"] or (arm != "bounded" and not match):
                raise ValueError(f"exact archive parity failed for {key}/{arm}")
            if result["labels"][0] != expected["labels"][0]:
                raise ValueError(f"first-window decision changed for {key}/{arm}")
            output[key] = dict(**result, rows=rows.tolist())
            if smoke:
                prefix = stream(template, eeg, rows[:31], arm)
                if prefix["labels"] != result["labels"][:31]:
                    raise ValueError(f"independent prefix or reset failed for {key}/{arm}")
                break
    if "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text():
        raise ValueError("worker unexpectedly mapped host dependencies")
    name = f"{arm}_{'smoke' if smoke else index}.json"
    evidence = dict(arm=arm, index=index, sessions=output,
                    source=hashlib.sha256(Path("source.sha256").read_bytes()).hexdigest(),
                    archive=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    path = Path(name)
    with path.open("x") as handle:
        json.dump(evidence, handle, allow_nan=False, sort_keys=True)
    with path.with_suffix(".sha256").open("x") as handle:
        handle.write(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    print(f"PASS F-C10 {arm}: {who}, {len(output)} exact worker sessions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("person", type=int, choices=range(9))
    parser.add_argument("arm", choices=("stock", "zero", "bounded"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.person, args.arm, args.smoke)
