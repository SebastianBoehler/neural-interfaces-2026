"""Compare the self-contained selector ZIP with its measured native source."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

FRIDAY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FRIDAY / path) for path in (
    "cap250_fbl2_package", "cap250c21_gate/native", "cap250c20e_shared",
)]
from smoke import digest, frames, fresh_model, load_paths
from build_gate import selector_source


def vector(pair, left, right, tolerance):
    np.testing.assert_allclose(left, right, rtol=0, atol=tolerance, equal_nan=False, err_msg=pair)
    return float(np.abs(left - right).max())


def exercise(reference_type, stock_template, packaged_template, name, windows, peer_window, tolerance):
    reference = reference_type(fresh_model(stock_template), audit=True)
    trial = fresh_model(packaged_template)
    protected = []
    stock_cp = trial.stock.cp
    baseline_method = stock_cp.step
    personalizer = type(stock_cp)
    original_step = personalizer.step

    def observed_terminal(instance, x, raw, count, recognized):
        result = original_step(instance, x, raw, count, recognized)
        if instance is stock_cp:
            protected.append(np.stack((raw, count, result)))
        return result
    drifts = dict.fromkeys(("stock", "alternate", "selected"), 0.)
    first = None
    with patch.object(personalizer, "step", observed_terminal):
        for frame in windows:
            a, b = reference.predict(frame[None]), trial.predict(frame[None])
            if not torch.equal(a, b) or b.shape != (1,) or b.dtype != torch.long or b.device.type != "cpu":
                raise ValueError("different official decision")
            first = int(b.item()) if first is None else first
            measured, direct = reference.outer, trial.selector.outer
            a_tail = np.stack((measured.raw[-1], measured.count[-1], measured.terminal[-1]))
            b_tail = np.stack((direct.raw[-1], direct.count[-1], direct.terminal[-1]))
            for key, left, right in (
                ("stock", reference.stock_stage[-1], protected[-1]), ("alternate", a_tail, b_tail),
                ("selected", reference.probability, trial.selector.probability),
            ):
                drifts[key] = max(drifts[key], vector(key, left, right, tolerance))
    original, packed = reference.summary(), trial.selector.summary()
    expected_rec, expected_ft = (10, 0) if name == "recognized" else (-1, 40)
    expected_eligible = 0 if name == "recognized" else 11
    work = ("rows", "recognized", "eligible", "alternate_refits", "alternate_fused", "alternate_fit_ages", "native_stop")
    assert all(original[key] == packed[key] for key in work)
    assert (packed["rows"], packed["recognized"], packed["eligible"]) == (31, expected_rec, expected_eligible)
    assert reference.model.ft.steps == trial.stock.ft.steps == expected_ft
    stock_cp.step = baseline_method

    active_copy = fresh_model(trial)
    trial.reset_state()
    trial.predict(peer_window)
    assert trial.selector.age == trial.stock.cp.t == 1
    trial.reset_state()
    for clean in (active_copy, trial, fresh_model(packaged_template)):
        assert clean.selector is None
        assert int(clean.predict(windows[:1]).item()) == first
        assert clean.selector.age == clean.stock.cp.t == 1
        vector("reset", clean.selector.probability, reference.selected[0], tolerance)
    return dict(rows=31, recognized=expected_rec, eligible=expected_eligible, optimizer_steps=expected_ft,
                drifts=drifts, resets=3, peer_interleaved=True)


def main(args):
    torch.set_num_threads(1)
    gpu = args.device == "cuda:0"
    if gpu:
        assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    namespace = {"np": np, "torch": torch}
    exec(compile(selector_source(), "<measured-native-selector>", "exec"), namespace)
    reference_type = namespace["LiveSelector"]
    receipts = {}
    with TemporaryDirectory() as scratch:
        stock, trial = load_paths(args.stock, args.trial, scratch, args.device)
        fixtures = tuple(frames(args.public, args.external, args.known, args.unknown))
        assert len(fixtures) == 2
        for index, (name, rows) in enumerate(fixtures):
            windows = rows.to(args.device)
            peer = fixtures[1-index][1][:1].to(args.device)
            if gpu:
                warm = fresh_model(stock)
                for frame in windows:
                    warm.predict(frame[None])
                del warm
            receipts[name] = exercise(reference_type, stock, trial, name, windows, peer, 5e-7 if gpu else 0.)
    print(json.dumps(dict(archive=digest(args.trial), device=args.device, labels_opened=False,
                          streams=receipts), sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("stock", "trial", "public", "external", "known", "unknown"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    main(parser.parse_args())
