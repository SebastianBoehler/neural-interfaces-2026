"""Load the lead executable on a verified fold and capture Team B encoder outputs."""
import argparse
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from folds import DATA_SHA, LEAD, LEAD_SHA, ORDERS, D, pinned_pair, streams
from verify import B, DEVICE, FOLDS, WEIGHTS, audit, digest, fold_fixture
from run import timed_load


def substitute(text, pattern, replacement):
    new, count = re.subn(pattern, lambda _match: replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected one source match for {pattern!r}, got {count}")
    return new


def fold_model(data, fold, work, source_patch=None):
    if digest(LEAD) != LEAD_SHA:
        raise ValueError("lead ZIP digest differs")
    weights = WEIGHTS / fold / "weights.pt"
    receipt = audit(data, fold, weights)
    pkg = work / "pkg"
    pkg.mkdir()
    with ZipFile(LEAD) as z:
        parent = Path(z.extract("submission.py", work))
    fold_fixture(pkg, parent, FOLDS / fold / "pkg", "hard")
    blob = B.bank(weights, 3)
    source = (pkg / "submission.py").read_text()
    source = substitute(source, r"^    TEAM_B_SHA = '[a-f0-9]{64}'$", f"    TEAM_B_SHA = {B.sha(blob)!r}")
    # Production reference posteriors are calibrated against the all-data weights.
    source = substitute(source, r"^            self\._check_team_b\(dev\)\n", "")
    if source_patch:
        source = source_patch(source)
    (pkg / "submission.py").write_text(source)
    (pkg / "team_b.pt").write_bytes(blob)
    model, _ = timed_load(pkg, data, DEVICE)
    assert [c.__name__ for c in type(model).__mro__[:2]] == ["TeamBlendStack", "HardWaveEnsemble"]
    assert model.routed == 1 and model.cf is not None and model.cp is not None
    assert model.team_b is not None and len(model.team_b.nets) == 9 and model.TEAM_B_MIX == 0.25
    return model, receipt["weights_sha"], B.sha(source.encode())


def stream(template, signals, rows, zero=False, capture=None):
    _, official = D.official()
    model = official.fresh_model(template)
    if zero:
        model.team_b = None
    elif capture is not None:
        original = model.team_b.posterior

        def tap(x):
            p = original(x)
            capture.append(p.cpu().numpy().copy())
            return p

        model.team_b.posterior = tap
    labels = []
    for row in rows:
        label = model.predict(signals[row:row + 1].to(DEVICE))
        assert label.shape == (1,) and label.dtype == torch.int64 and label.item() in (0, 1, 2)
        labels.append(label.item())
    assert not model.dead and model.log6["err"] == 0 and model.log6["v6"] == len(rows), model.log6
    assert not model.cp.dead and not model.cf.dead and model.cp.log["err"] == model.cf.log["err"] == 0
    return np.asarray(labels, dtype=np.int8)


def encoder_stream(template, signals, rows):
    _, official = D.official()
    model = official.fresh_model(template)
    if model.perm is not None or not torch.isfinite(signals[rows]).all():
        raise ValueError("encoder diagnostic expects the pinned finite tensor in native channel order")
    post = [model.team_b.posterior(signals[row]).cpu().numpy() for row in rows]
    assert not template.team_b.windows and len(model.team_b.windows) == len(rows)
    assert all(g.graph is not None and not g.failed for g in model.team_b.feature_graphs.values())
    post = np.asarray(post)
    assert post.shape == (len(rows), 3) and np.isfinite(post).all()
    np.testing.assert_allclose(post.sum(1), 1, atol=1e-6, rtol=0)
    return post


def smoke(data, signals, model, weights):
    order, fold, stream_key = ORDERS[0], "new_P01", "P01|1"
    rows = D.plans(data, order, [stream_key])[stream_key]
    base, lead, expected_weights = pinned_pair(fold, order, stream_key, rows)
    assert weights == expected_weights
    taps = []
    np.testing.assert_array_equal(stream(model, signals, rows, zero=True), base)
    np.testing.assert_array_equal(stream(model, signals, rows, capture=taps), lead)
    np.testing.assert_allclose(encoder_stream(model, signals, rows), taps, atol=1e-7, rtol=0)
    print(f"PASS pinned lead and blend-zero on {len(rows)} windows, encoder equals live taps", flush=True)


def encode(data, signals, model, fold, out, weights, source):
    result = dict(weights_sha=weights, source_sha=source)
    for order in ORDERS:
        for stream_key, rows in D.plans(data, order, streams(data, fold)).items():
            _, _, expected_weights = pinned_pair(fold, order, stream_key, rows)
            assert weights == expected_weights
            suf = f"{order}_{stream_key.split('|')[1]}"
            result[f"rows_{suf}"] = rows
            result[f"post_{suf}"] = encoder_stream(model, signals, rows)
    np.savez_compressed(out / f"{fold}.npz", **result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fold", default="new_P01")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name()
    if digest(args.data) != DATA_SHA:
        raise ValueError("unexpected EEG data digest")
    data = D.load(args.data)
    signals = torch.from_numpy(data["X"])
    with TemporaryDirectory() as folder:
        model, weights, source = fold_model(data, args.fold, Path(folder))
        if args.smoke:
            smoke(data, signals, model, weights)
        else:
            encode(data, signals, model, args.fold, args.out, weights, source)


if __name__ == "__main__":
    main()
