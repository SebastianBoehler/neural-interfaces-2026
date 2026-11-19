"""Authenticate the read-only Eye fit receipts, masks and TorchScript traces."""

import argparse
import json

from eye_scope import FIT, FOLDS, OCUNET, PEOPLE, PRODUCER, pin, sha


def check(who, seed, smoke, source):
    directory = FIT / ("smoke" if smoke else "fits") / who
    trace = directory / f"eye_s{seed}.pt"
    receipt = trace.with_suffix(".json")
    p = json.loads(receipt.read_text())
    _, n, h, own, view, fit_sha, held_sha = FOLDS[who]
    epoch = 1 if smoke else 60
    expected = dict(fold=who, member="eye", seed=seed, smoke=smoke, epochs=epoch, updates=36 * epoch,
                    source=source, basis=view, fit_rows_sha256=fit_sha, held_rows_sha256=held_sha,
                    n_source=n, n_held=h, n_own_visit1=own, actual_source_loads=1, original_recipe=sha(PRODUCER),
                    ocular_recipe=sha(OCUNET), held_labels_opened=False, held_waveforms_used_in_fit=False)
    tests = p["tests"]
    if (any(p[k] != v for k, v in expected.items()) or not 0 <= p["export_max_abs"] < 1e-5
            or not 0 <= tests["exported_batch_single_max_abs"] < 1e-5
            or not all(tests[k] for k in ("prefix_exact", "interleaved_reset_exact", "first_window_exact"))
            or tests["source_visits"] != 12 or "A100" not in p["gpu"]
            or any(a.stat().st_mode & 0o222 for a in (receipt, trace))):
        raise ValueError(f"unauthenticated real Eye {'smoke' if smoke else 'fit'} {who}/{seed}")
    pin(trace, p["trace"])
    return dict(receipt=sha(receipt), path=str(trace), trace=p["trace"], updates=p["updates"])


def artifacts(who, source):
    pin(FIT / "fit_source.sha256", source)
    smoke = {name: check(name, 0, True, source) for name in PEOPLE}
    fits = [check(who, seed, False, source) for seed in range(3)]
    if len({f["trace"] for f in fits}) != 3:
        raise ValueError("three independent actual Eye seed traces are required")
    return fits, smoke


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("smoke",))
    cli.parse_args()
    source = sha(FIT / "fit_source.sha256")
    for who in PEOPLE:
        print(check(who, 0, True, source), flush=True)
