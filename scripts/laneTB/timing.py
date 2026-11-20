"""Measure paired hardcb overhead and Team B encoding on a warmed A100."""
import argparse
import json
import time
from pathlib import Path

import torch

from run import D, DEVICE, stream_session, sync, time_encoder, timed_load


def components(template, signals, rows):
    _, official = D.official()
    model = official.fresh_model(template)

    def forward(row):
        raw = signals[row].numpy()
        model.hard_probability(raw)
        model.cb.prob(raw)

    forward(rows[0])
    sync(DEVICE)
    start = time.perf_counter()
    for row in rows[1:]:
        forward(row)
    sync(DEVICE)
    return 1000 * (time.perf_counter() - start) / (len(rows) - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "run", "data"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    smoke = a.run / "streams/smoke"
    receipt = json.loads((smoke / "smoke.json").read_text())
    d = D.load(a.data)
    x = torch.from_numpy(d["X"])
    rows = D.plans(d, 71000, ["P01|1"])["P01|1"]
    assert len(rows) == receipt["windows"]
    packages = {
        "stock": a.source / "stock/new_P01/pkg",
        "safe": a.source.parent / "overnight-21nov-pkgcmp/hardcb_safe/new_P01/pkg",
        "nine15": smoke / "nine15/pkg",
        "three15": smoke / "three15/pkg",
    }
    models = {k: timed_load(p, d, DEVICE)[0] for k, p in packages.items()}
    for arm in ("stock", "safe"):
        stream_session(models[arm], x, rows, DEVICE)
    for arm in ("nine15", "three15"):
        stream_session(models[arm], x, rows[:68], DEVICE)
        time_encoder(models[arm].team_b, x, rows, DEVICE)

    results = {k: dict(load_s=[], runs=[], encoder_ms=[]) for k in models}
    for order in (list(models), list(reversed(models))):
        for arm in order:
            model, elapsed = timed_load(packages[arm], d, DEVICE)
            results[arm]["load_s"].append(elapsed)
            del model
            used_rows = rows if arm in {"stock", "safe"} else rows[:68]
            _, _, meta = stream_session(models[arm], x, used_rows, DEVICE)
            results[arm]["runs"].append(meta)
            if arm in {"nine15", "three15"}:
                results[arm]["encoder_ms"].append(time_encoder(models[arm].team_b, x, rows, DEVICE))
            print(arm, elapsed, meta, flush=True)
    data = dict(device=torch.cuda.get_device_name(), windows=len(rows), warmup_windows=len(rows),
                timings=results, hardcb_component_ms=components(models["safe"], x, rows),
                smoke_posterior_drift=receipt["zero_posterior_drift"])
    scratch = a.run / ".timing.json"
    scratch.write_text(json.dumps(data, indent=2) + "\n")
    scratch.rename(a.run / "timing.json")


if __name__ == "__main__":
    main()
