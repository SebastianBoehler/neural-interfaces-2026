"""Authenticate the joint-excluded Eye actor before calibration or use."""

import json

from eye_scope import OCUNET, PEOPLE, PRODUCER, pin, rows_sha, sha
from joint_eye import BATCHES, FIT, scope


def verify(seed, smoke, source):
    idx, held, _, _ = scope()
    path = FIT / ("smoke" if smoke else "fits") / f"eye_s{seed}.pt"
    receipt = path.with_suffix(".json")
    p = json.loads(receipt.read_text())
    epoch = 1 if smoke else 60
    expect = dict(source=source, member="eye", seed=seed, smoke=smoke, epochs=epoch, updates=BATCHES * epoch,
                  source_loads=1, recipe=sha(PRODUCER), ocular_features=sha(OCUNET), train_rows=rows_sha(idx),
                  excluded_rows=rows_sha(held), n_source=len(idx), n_excluded=len(held), excluded_people=list(PEOPLE),
                  excluded_waveforms_used_in_fit=False, excluded_labels_opened=False)
    checks = p["tests"]
    if (any(p[k] != v for k, v in expect.items()) or not 0 <= p["export_max_abs"] < 1e-5
            or not 0 <= checks["exported_batch_single_max_abs"] < 1e-5 or "A100" not in p["backend"]
            or checks["source_visits"] != 8
            or not all(checks[k] for k in ("prefix_exact", "interleaved_reset_exact", "first_window_exact"))
            or any(f.stat().st_mode & 0o222 for f in (receipt, path))):
        raise ValueError(f"incomplete source-only Eye fit proof {seed, smoke}")
    pin(path, p["trace"])
    return dict(receipt=sha(receipt), trace=p["trace"], path=str(path))


def all_fits(source):
    pin(FIT / "fit_source.sha256", source)
    smoke = verify(0, True, source)
    actual = [verify(seed, False, source) for seed in range(3)]
    if len({x["trace"] for x in actual}) != 3:
        raise ValueError("not three distinct trace bytes")
    return actual, smoke


if __name__ == "__main__":
    print(verify(0, True, sha(FIT / "fit_source.sha256")), flush=True)
