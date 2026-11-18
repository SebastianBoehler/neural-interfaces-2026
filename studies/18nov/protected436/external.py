"""Full raw external streams with actual gates, checked against exact published cap250 outputs."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from data import permutation


def run(template, x, rows, control):
    model = fresh_model(template)
    labels = []
    base_step = model.base_personal.step
    observed = []

    def observe(*args):
        result = base_step(*args)
        observed.append(int(result.argmax()))
        return result

    model.base_personal.step = observe
    for r in rows:
        label = model.predict(x[r:r+1].to("cuda:0"))
        if label.shape != (1,) or label.dtype != torch.int64 or label.device.type != "cpu":
            raise ValueError("unexpected worker label")
        labels.append(int(label[0]))
    if observed != control:
        raise ValueError(f"protected baseline and exact §385 differ on {sum(a!=b for a,b in zip(observed,control))} rows")
    if model.ft.over or model.dead or len(observed) != len(rows):
        raise ValueError("incomplete external policy")
    return dict(rows=rows.tolist(), labels=labels, activity=replay.activity(model))


def main(person):
    if torch.__version__ != "2.14.1+cu130" or "A100" not in torch.cuda.get_device_name():
        raise ValueError("expected §385 worker")
    with np.load("/external.npz") as z:
        x, people, days, names = torch.from_numpy(z["X"]), z["subject"], z["session"], z["ch_names"].tolist()
    streams = np.char.add(np.char.add(people, "|"), days)
    i = max(person, 0)
    who = sorted(set(people))[i]
    controls = json.loads(Path(f"/prior/cap_{i}.json").read_text())
    template = replay.load(Path("/work/protected"), names)
    entries = {}
    for seed in (0, 1):
        order = permutation(streams, None, "full", seed)
        for day in sorted(set(days[people == who])):
            key = f"{who}|{day}|{seed}"
            rows = order[(people == who) & (days == day)]
            if not np.array_equal(rows, controls[key]["rows"]):
                raise ValueError("pinned order differs")
            if person == -1:
                rows = rows[:96]
            controls_here = controls[key]["labels"][:len(rows)]
            entries[key] = run(template, x, rows, controls_here)
            if person == -1:
                if run(template, x, rows[:17], controls_here[:17])["labels"] != entries[key]["labels"][:17]:
                    raise ValueError("prefix reset failed")
                break
        if person == -1:
            break
    Path(f"policy_{person}.json").write_text(json.dumps(entries))


if __name__ == "__main__":
    main(int(sys.argv[1]))
