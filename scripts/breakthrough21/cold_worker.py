"""Replay the population covariance prior on the exact fused Team B folds."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from worker import DEVICE, D, digest, load, stream, timed_load
from fit import DATA_SHA, roster
from fixture import ORDER_PIN, TEAM_B_PIN, stage, visits
from patch_cold import patch
from patch_terminal import verify_parent
from terminal_collect import contrast, line, load as collect

SCREEN = tuple(range(71624, 71632))
HELD = tuple(range(71632, 71648))


def candidate(run, fold, data, out):
    receipt = stage(run, data, fold, out, (), observe=False)
    p = run / "priors" / f"{fold}.npy"
    info = json.loads(p.with_suffix(".json").read_text())
    rows, people, first = roster(fold, data)
    expected = dict(data_sha=DATA_SHA, fold=fold, people=people, first_only=first,
                    streams=sorted(set(data["stream"][rows])), rows=len(rows), prior_sha=digest(p))
    if info != expected:
        raise ValueError(f"unexpected source rows or prior for {fold}")
    prior = np.load(p, allow_pickle=False)
    path = out / "submission.py"
    path.write_text(patch(path.read_text(), prior))
    model, _ = timed_load(out, data, DEVICE)
    torch.testing.assert_close(model.team_b.population_cov.cpu(), torch.from_numpy(prior), atol=0, rtol=0)
    if model.team_b._lookup_prior(1) != 12 or model.team_b._lookup_prior(40) <= 0 or model.team_b._lookup_prior(41) != 0:
        raise RuntimeError("population prior schedule is inactive or leaks past the first 40 windows")
    return model, dict(receipt, arms=["cold"], candidate_sha=digest(path), prior=expected)


def smoke(run, data, signals, fold, tmp):
    s = visits(data, fold)[0]
    rows = D.plans(data, ORDER_PIN, [s])[s]
    reference, _ = load(run, fold, data, tmp / "lead", (), observe=False)
    cold, _ = candidate(run, fold, data, tmp / "cold")
    prior = cold.team_b.population_cov
    cold.team_b.population_cov = None
    lead, _ = stream(reference, signals, rows, (), observe=False)
    zero, _ = stream(cold, signals, rows, (), observe=False)
    np.testing.assert_array_equal(lead["lead"], zero["lead"])
    key = f"{ORDER_PIN}_{s.split('|')[1]}"
    with np.load(TEAM_B_PIN / fold / "nine25" / f"pred_{fold}_{ORDER_PIN}.npz", allow_pickle=False) as pin:
        np.testing.assert_array_equal(pin[f"rows_{key}"], rows)
        np.testing.assert_array_equal(pin[f"labels_{key}"], lead["lead"])
    cold.team_b.population_cov = prior
    x = signals[rows[0]].to(DEVICE)
    t0 = reference.team_b.posterior(x)
    t1 = cold.team_b.posterior(x)
    assert (t0 - t1).abs().max().item() > 1e-4
    reference.team_b.reset()
    cold.team_b.reset()
    full, _ = stream(cold, signals, rows, (), observe=False)
    prefix, _ = stream(cold, signals, rows[:64], (), observe=False)
    np.testing.assert_array_equal(full["lead"][:64], prefix["lead"])
    print(f"PASS {fold}: {len(rows)} pinned/zero labels, active prefix prior and 64-label reset", flush=True)


def replay(run, data, signals, fold, tmp, block, chunk):
    if block == "cold_held" and not json.loads((run / "cold_screen.json").read_text())["advance"]:
        print("Cold screen did not advance. Held-out orders remain unopened.")
        return
    orders = (SCREEN if block == "cold_screen" else HELD)[2 * chunk:2 * chunk + 2]
    if len(orders) != 2:
        raise ValueError("missing cold order chunk")
    reference, _ = load(run, fold, data, tmp / "lead", (), observe=False)
    cold, receipt = candidate(run, fold, data, tmp / "cold")
    out = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    for o in orders:
        for s, rows in D.plans(data, o, visits(data, fold)).items():
            key = f"{o}_{s.split('|')[1]}"
            out[f"rows_{key}"] = rows
            for arm, template in (("lead", reference), ("cold", cold)):
                pred, _ = stream(template, signals, rows, (), observe=False)
                out[f"{arm}_{key}"] = pred["lead"]
            print(f"PASS {fold} {o} {s} {len(rows)}", flush=True)
    target = run / block / fold
    target.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target) as scratch:
        path = Path(scratch) / f"{orders[0]}.npz"
        np.savez_compressed(path, **out)
        path.rename(target / path.name)


def report(run, data, block):
    if block == "cold_held" and not json.loads((run / "cold_screen.json").read_text())["advance"]:
        print("Cold screen did not advance.")
        return
    orders = SCREEN if block == "cold_screen" else HELD
    scores, bins, receipts = collect(run, block, ("lead", "cold"), orders, data)
    result = contrast(scores["cold"], scores["lead"], orders)
    first_40 = bins["new"]["cold"][-1]
    eligible = result["new"]["mean"] > 0 and result["returning"]["mean"] >= -0.30 and first_40 > 0
    print(line("cold", result), f"first 40 new {first_40:+.2f} pp", flush=True)
    with (run / f"{block}.json").open("x") as f:
        json.dump(dict(orders=orders, advance=bool(eligible), delta=result, delta_by_delivery_position=bins,
                       first_40_new_pp=float(first_40), fold_receipts=receipts), f, indent=2)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("smoke", "cold_screen", "cold_held", "report"))
    p.add_argument("run", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("fold_or_block")
    p.add_argument("chunk", type=int, nargs="?")
    a = p.parse_args()
    verify_parent(a.data, DATA_SHA)
    data = D.load(a.data)
    if a.action == "report":
        report(a.run, data, a.fold_or_block)
        return
    signals = torch.from_numpy(data["X"])
    with TemporaryDirectory() as tmp:
        if a.action == "smoke":
            smoke(a.run, data, signals, a.fold_or_block, Path(tmp))
        else:
            replay(a.run, data, signals, a.fold_or_block, Path(tmp), a.action, a.chunk)


if __name__ == "__main__":
    main()
