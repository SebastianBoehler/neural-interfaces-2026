"""Run either original package against identical EEG without accessing task labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import replay
from data import permutation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["native", "python"])
    parser.add_argument("person", type=int)
    parser.add_argument("suite", choices=["smoke", "full"])
    args = parser.parse_args()
    assert torch.__version__ == "2.14.1+cu130" and "A100" in torch.cuda.get_device_name()
    with np.load("/external.npz") as source:
        x = torch.from_numpy(source["X"])
        people, days = source["subject"], source["session"]
        channels = source["ch_names"].tolist()
    assert x.shape == (2130, 41, 480) and x.dtype == torch.float32
    subject = sorted(set(people))[args.person]
    archive = json.loads(Path(f"/archive/cap_{args.person}.json").read_text())
    model = replay.load(Path("/package"), channels)
    assert hasattr(model, "handle") == (args.arm == "native")
    streams = np.char.add(np.char.add(people, "|"), days)
    result = {}
    for seed in (0, 1):
        order = permutation(streams, None, "full", seed)
        for day in sorted(set(days[people == subject])):
            name = f"{subject}|{day}|{seed}"
            rows = order[(people == subject) & (days == day)]
            assert rows.tolist() == archive[name]["rows"]
            record = replay.stream(model, x, rows)
            activity = record["activity"]
            assert activity["recognized"] == -1 and activity["ft_refits"] > 0
            assert activity["cp_fused"] > 0 and activity["ft_over"] == 0
            result[name] = dict(record, rows=rows.tolist())
            if args.suite == "smoke":
                break
        if args.suite == "smoke":
            break
    if args.suite == "full" and args.person in (4, 5):
        name = next(iter(result))
        rows = result[name]["rows"]
        result["repeat"] = dict(replay.stream(model, x, rows), rows=rows)
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    Path(f"{args.suite}_{args.arm}_{args.person}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
