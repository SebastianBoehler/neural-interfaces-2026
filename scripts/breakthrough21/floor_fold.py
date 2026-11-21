"""Paired full-Solver replay of the fused lead and audited Team A terminal floor."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from worker import D, DEVICE, digest, load, stream, timed_load
from build_ba_terminal import once
from check_ba_terminal import play
from fit import DATA_SHA
from fixture import ORDER_PIN, TEAM_B_PIN, stage, visits
from grow_sources import team_a_donor
from patch_team_b_bank import with_manas
from patch_terminal import verify_parent
from terminal_collect import contrast, line, load as collect

ORDERS = tuple(range(71648, 71656))
TEAM_B_TEST = "            self._check_team_b(dev)\n"
TEAM_B_LOAD = '            self.team_b = _TEAM_B["Encoder"](pack, dev, self.TEAM_B_MEMBERS)\n'


def candidate(run, fold, data, out):
    receipt = stage(run, data, fold, out, (), observe=False)
    donor, bank = team_a_donor(fold, data, DATA_SHA)
    source = out / "submission.py"
    text = once(source.read_text(), TEAM_B_LOAD, TEAM_B_LOAD + TEAM_B_TEST)
    new = with_manas(text, donor, bank["weights_sha"], (donor / "LICENSE-MANAS.txt").read_text())
    source.write_text(once(new, TEAM_B_TEST, ""))
    (out / "team_a.pt").symlink_to(donor / "weights.pt")
    model, _ = timed_load(out, data, DEVICE)
    if (type(model).__name__ != "TerminalTeamAStack" or model.TEAM_A_MIX != 0.10 or
            not model.TEAM_A_GRAPH or model.team_a is None):
        raise ValueError("inactive MANAS floor")
    return model, dict(receipt, arms=["team_a"], candidate_sha=digest(source), manas=bank)


def smoke(data, signals, fold, lead, floor):
    s = visits(data, fold)[0]
    rows = D.plans(data, ORDER_PIN, [s])[s]
    expected, _ = stream(lead, signals, rows, (), observe=False)
    suffix = f"{ORDER_PIN}_{s.split('|')[1]}"
    with np.load(TEAM_B_PIN / fold / "nine25" / f"pred_{fold}_{ORDER_PIN}.npz", allow_pickle=False) as pin:
        np.testing.assert_array_equal(rows, pin[f"rows_{suffix}"])
        np.testing.assert_array_equal(expected["lead"], pin[f"labels_{suffix}"])
    windows = signals[rows].to(DEVICE)
    labels, _, _ = play(floor, windows, capture=True)
    prefix, _, _ = play(floor, windows[:32], capture=True)
    np.testing.assert_array_equal(labels[:32], prefix)
    print(f"PASS {fold}: {len(rows)} pinned lead and live Team A terminal formula, fresh prefix 32", flush=True)


def replay(a, data, signals, lead, floor, receipt):
    orders = ORDERS[2 * a.chunk:2 * a.chunk + 2]
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    for o in orders:
        for s, rows in D.plans(data, o, visits(data, a.fold)).items():
            key = f"{o}_{s.split('|')[1]}"
            out, _ = stream(lead, signals, rows, (), observe=False)
            label, _, _ = play(floor, signals[rows].to(DEVICE))
            payload.update({f"rows_{key}": rows, f"lead_{key}": out["lead"], f"team_a_{key}": label.astype(np.int8)})
            print(f"PASS {a.fold} {o} {s} {len(rows)}", flush=True)
    target = a.run / "floor_held" / a.fold
    target.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target) as tmp:
        path = Path(tmp) / f"{orders[0]}.npz"
        np.savez_compressed(path, **payload)
        path.rename(target / path.name)


def report(run, data):
    scores, bins, receipts = collect(run, "floor_held", ("lead", "team_a"), ORDERS, data)
    delta = contrast(scores["team_a"], scores["lead"], ORDERS)
    result = dict(orders=ORDERS, delta=delta, delta_by_delivery_position=bins, fold_receipts=receipts)
    with (run / "floor_held.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(line("team_a", delta), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("smoke", "score", "report"))
    p.add_argument("run", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("fold")
    p.add_argument("chunk", type=int)
    a = p.parse_args()
    verify_parent(a.data, DATA_SHA)
    data = D.load(a.data)
    if a.action == "report":
        report(a.run, data)
        return
    signals = torch.from_numpy(data["X"])
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        lead, _ = load(a.run, a.fold, data, root / "lead", (), observe=False)
        floor, receipt = candidate(a.run, a.fold, data, root / "floor")
        if a.action == "smoke":
            smoke(data, signals, a.fold, lead, floor)
        else:
            replay(a, data, signals, lead, floor, receipt)


if __name__ == "__main__":
    main()
