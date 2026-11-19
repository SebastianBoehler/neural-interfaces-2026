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
from smoke import digest, frames, fresh_model, load_paths, pinned

REFERENCE_SHA = "84cbe539440f040dcb2ce3f74642313eaf3c8dee6bef7dd44aeff06dee691b45"


def vector(pair, left, right, tolerance):
    np.testing.assert_allclose(left, right, rtol=0, atol=tolerance, equal_nan=False, err_msg=pair)
    return float(np.abs(left - right).max())


def exercise(reference_type, stock_template, packaged_template, name, windows, peer_window, tolerance, optimized):
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
        for age, frame in enumerate(windows):
            a, b = reference.predict(frame[None]), trial.predict(frame[None])
            if not torch.equal(a, b) or b.shape != (1,) or b.dtype != torch.long or b.device.type != "cpu":
                raise ValueError("different official decision")
            first = int(b.item()) if first is None else first
            measured, direct = reference.outer, trial.selector.outer
            for key, left, right in (
                ("stock", reference.stock_stage[-1], protected[-1]),
                ("selected", reference.probability, trial.selector.probability),
            ):
                drifts[key] = max(drifts[key], vector(key, left, right, tolerance))
            if len(direct.raw) == age + 1:
                a_tail = np.stack((measured.raw[-1], measured.count[-1], measured.terminal[-1]))
                b_tail = np.stack((direct.raw[-1], direct.count[-1], direct.terminal[-1]))
                drifts["alternate"] = max(drifts["alternate"], vector("alternate", a_tail, b_tail, tolerance))
            elif not (optimized and trial.stock.recB and len(direct.raw) == trial.selector.recognition_age):
                raise ValueError("unexpected early stop of the alternate trajectory")
    original, packed = reference.summary(), trial.selector.summary()
    length = len(windows)
    recognized = name == "recognized"
    expected_rec = 10 if recognized else -1
    expected_ft = 0 if recognized else 20 * len(range(20, length, 10))
    expected_eligible = 0 if recognized else length - 20
    work = ("rows", "recognized", "eligible", "alternate_refits", "alternate_fused", "alternate_fit_ages", "native_stop")
    assert all(original[key] == packed[key] for key in work)
    assert (packed["rows"], packed["recognized"], packed["eligible"]) == (length, expected_rec, expected_eligible)
    if optimized:
        private_rows = expected_rec if recognized else length
        assert packed["alternate_rows"] == packed["shared_calls"] == private_rows
        assert packed["retired_at"] == (expected_rec if recognized else None)
        expected_work = (0, 0) if recognized else (5, 30)
        assert (packed["alternate_refits"], packed["alternate_fused"]) == expected_work
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
    return dict(rows=length, recognized=expected_rec, eligible=expected_eligible, optimizer_steps=expected_ft,
                drifts=drifts, resets=3, peer_interleaved=True, shared=packed.get("shared_calls"),
                alternate_refits=packed["alternate_refits"], alternate_fused=packed["alternate_fused"])


def main(args):
    torch.set_num_threads(1)
    gpu = args.device == "cuda:0"
    if gpu:
        assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    namespace = {"np": np, "torch": torch}
    exec(compile(pinned(args.reference, REFERENCE_SHA).read_text(), "<measured-native-selector>", "exec"), namespace)
    reference_type = namespace["LiveSelector"]
    receipts = {}
    with TemporaryDirectory() as scratch:
        stock, trial = load_paths(args.stock, args.trial, scratch, args.device)
        fixtures = tuple(frames(args.public, args.external, args.known, args.unknown,
                                external_rows=120 if args.optimized else 31))
        assert len(fixtures) == 2
        for index, (name, rows) in enumerate(fixtures):
            windows = rows.to(args.device)
            peer = fixtures[1-index][1][:1].to(args.device)
            if gpu:
                warm = fresh_model(stock)
                for age, frame in enumerate(windows):
                    warm.predict(frame[None])
                del warm
            receipts[name] = exercise(reference_type, stock, trial, name, windows, peer, 5e-5 if gpu else 0., args.optimized)
    print(json.dumps(dict(archive=digest(args.trial), device=args.device, labels_opened=False,
                          streams=receipts), sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("stock", "trial", "public", "external", "known", "unknown", "reference"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    main(parser.parse_args())
