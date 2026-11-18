"""Replay one unmodified release on an RTX 2080 Ti with no task labels."""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from data import permutation


def stream(template, eeg, rows, arm):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    created = perf_counter()
    labels = []
    for row in rows:
        window = eeg[int(row):int(row)+1].to("cuda:0")
        label = model.predict(window)
        assert label.device.type == "cpu" and label.shape == (1,) and label.dtype == torch.int64
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    activity = replay.activity(model) if arm == "cap" else dict(model.route.log)
    if arm == "cap":
        assert activity["windows"] == len(rows) and not activity["errors"] and not model.dead
    else:
        assert activity["n"] == len(rows) and not activity["err"]
    assert set(labels) <= {0, 1, 2}
    return dict(labels=labels, rows=rows.tolist(), activity=activity,
                seconds=perf_counter() - start, copy_seconds=created - start)


def main(arm, person, smoke):
    device = torch.cuda.get_device_name()
    assert "RTX 2080 Ti" in device and torch.__version__ == "2.14.1+cu130"
    with np.load("/external.npz", allow_pickle=False) as source:
        eeg = torch.from_numpy(source["X"])
        people, days = source["subject"], source["session"]
        channels = source["ch_names"].tolist()
    assert eeg.shape == (2130, 41, 480) and eeg.dtype == torch.float32 and len(channels) == 41
    subjects = sorted(set(people))
    who = subjects[person]
    assert (person, who) in ((0, "Scherer2015/A"), (5, "Scherer2015/G"))
    sessions = np.char.add(np.char.add(people, "|"), days)
    reference = json.loads((Path("/prior") / f"{arm}_{person}.json").read_text())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    template = replay.load(Path("/package"), channels)
    torch.cuda.synchronize()
    load_seconds = perf_counter() - start
    results = {}
    for order in ((0,) if smoke else (0, 1)):
        perm = permutation(sessions, None, "full", order)
        visits = sorted(set(days[people == who]))
        for day in (visits[:1] if smoke else visits):
            key = f"{who}|{day}|{order}"
            rows = perm[(people == who) & (days == day)]
            expected = reference[key]
            assert rows.tolist() == expected["rows"]
            if smoke:
                rows = rows[:42]
            row = stream(template, eeg, rows, arm)
            results[key] = row
            if smoke:
                prefix = stream(template, eeg, rows[:17], arm)
                assert prefix["labels"] == row["labels"][:17]
                if arm == "cap":
                    activity = row["activity"]
                    assert activity["recognized"] == -1
                    assert activity["ft_refits"] > 0 and activity["ft_steps"] > 0
            else:
                assert row["activity"] == expected["activity"], key
            changed = sum(a != b for a, b in zip(row["labels"], expected["labels"]))
            print(key, f"{row['seconds']:.3f} seconds", f"{changed} label differences", flush=True)
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    result = dict(arm=arm, person=person, device=device, capability=torch.cuda.get_device_capability(),
                  torch=torch.__version__, load_seconds=load_seconds,
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), sessions=results)
    suffix = "smoke" if smoke else str(person)
    Path(f"rtx_{arm}_{suffix}.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:] == ["smoke"])
