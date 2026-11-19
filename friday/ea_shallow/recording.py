"""Verify one historical ShallowFBCSPNet fit against its archived posteriors."""
import argparse
import json
import os
from pathlib import Path

import braindecode
import deep
import features
import numpy as np
import probe
import protocol
import run_cv
import torch
from source_labels import read_training_labels

HERE = Path(__file__).resolve().parent
REV = "c625ecc736a881836a25f85a741ea9147bf698bf"
SOURCE = "32612c9c02caaa8adf74dbedeed566bcfec7c9ad9e2a1b6e7d21731cefe1313c"
ARCHIVE = "f4169ebbc0fe7ad51c55b18c6f2ded5b9223b467086bfed190daf9b641385385"
PRODUCER = "1658ce9c362630ebd4e7398efb5999a4421673cf765c70362cc3a66b1af2d925"
MODEL, HELD, SEED = "shallow_ea", "P01", 5


def run(root, phase):
    for module in (deep, features, protocol, run_cv):
        if Path(module.__file__).resolve().parent != probe.HERE / REV / "experiments":
            raise ValueError(f"unexpected source import {module.__name__}")
    if probe.digest(probe.HERE / "source.sha256") != SOURCE or probe.digest(probe.HERE / "archive.sha256") != ARCHIVE:
        raise ValueError("unexpected borrowed proof")
    producer = root / "runs/experiments/results-3002618-5.jsonl"
    if probe.digest(producer) != PRODUCER:
        raise ValueError("historical producer receipt changed")
    if (torch.__version__, braindecode.__version__, np.__version__) != ("2.14.1+cu126", "1.8.1", "2.5.4"):
        raise ValueError("unexpected historical runtime versions")

    target = probe.historical(root)[MODEL][SEED]
    data, full0 = probe.read_inputs(root)
    held_subject = f"Dreyer2026Proteus/{HELD}"
    _, tr, te = next(fold for fold in protocol.folds(data, "all_loso") if fold[0] == HELD)
    fit, test = np.flatnonzero(tr), np.flatnonzero(te)
    if not np.array_equal(tr, data["subject"] != held_subject) or not np.array_equal(te, ~tr):
        raise ValueError("historical source fold differs")
    rows = full0[te[full0]]
    if (len(fit), len(test), len(set(data["stream"][test]))) != (2465, 190, 1):
        raise ValueError("unexpected source or held row counts")
    if not np.array_equal(np.sort(rows), test):
        raise ValueError("delivery rows omit held windows")

    run_cv.SEED[0] = SEED
    actor = run_cv.CONFIGS[MODEL](data)
    if (actor.arch, actor.align, actor.k, actor.epochs, actor.lr, actor.wd, actor.batch, actor.seed) != (
        "shallow", True, 5.0, 40, 1e-3, 1e-3, 64, SEED
    ) or (actor.adabn, actor.aug, actor.train_fir, actor.ft, actor.ls, actor.lld, actor.lp, actor.crops, actor.net_kw) != (
        None, False, False, 0, 0.0, 1.0, 0, (), {}
    ):
        raise ValueError("unexpected historical shallow recipe")
    prior = features.covariances(data["X"][fit]).mean(0)
    if not np.array_equal(prior, actor.C[fit].mean(0)):
        raise ValueError("source alignment prior differs")
    actor.G = prior.copy()
    first = probe.first_window(actor, features, test, rows)
    if first["first_row"] != 94 or not np.isclose(first["rec_lookup_max_abs"], 3.072798252105713):
        raise ValueError("historical P01 first-window control differs")
    receipt = dict(model=MODEL, fold=HELD, seed=SEED, revision=REV, source=SOURCE, archive=ARCHIVE,
                   producer=PRODUCER, held_labels_opened=False, prior=probe.fingerprint(prior),
                   recording_archive=probe.fingerprint(target[test]), input=first,
                   fit_rows=probe.fingerprint(fit), held_rows=probe.fingerprint(test))
    input_receipt = HERE / "P01_seed5_inputs.json"
    if phase == "inputs":
        probe.write_json(input_receipt, receipt)
        return
    if json.loads(input_receipt.read_text()) != receipt:
        raise ValueError("label-free archive input proof differs")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name() or actor.dev != "cuda":
        raise ValueError("historical A100 backend required")
    if torch.version.cuda != "12.6" or torch.backends.cudnn.version() != 91002:
        raise ValueError("unexpected historical CUDA/cuDNN versions")

    data["y"] = np.full(len(tr), -1, np.int64)
    data["y"][fit] = read_training_labels(root / "data/proteus_windows.npz", fit, len(tr))
    data["w"] = np.ones(len(tr))
    actor.fit(fit)
    if (data["y"][test] != -1).any() or not np.array_equal(actor.G, prior) or actor.net.training:
        raise ValueError("source fit changed exclusion, prior or inference mode")
    if type(actor.net) is not braindecode.models.ShallowFBCSPNet:
        raise ValueError("unexpected fitted model class")

    rec = protocol.predict_streams(actor, data, te, proba=True)
    expected = target[test]
    if rec.shape != expected.shape or not np.isfinite(rec).all():
        raise ValueError("invalid recording posteriors")
    changes = int(np.count_nonzero(rec.argmax(1) != expected.argmax(1)))
    error = float(np.abs(rec - expected).max())
    parity = changes == 0 and error <= probe.TOL
    receipt.update(recording_decision_changes=changes, recording_max_abs=error,
                   parity=parity, bit_exact=np.array_equal(rec, expected), n_held=len(test),
                   fitted_state=probe.tensor_identity(actor.net), gpu=torch.cuda.get_device_name(),
                   node=os.environ["SLURMD_NODENAME"], cudnn=torch.backends.cudnn.version(),
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   cudnn_deterministic=torch.backends.cudnn.deterministic,
                   cudnn_benchmark=torch.backends.cudnn.benchmark,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    probe.write_json(HERE / "P01_seed5_recording.json", receipt)
    if not parity:
        raise ValueError(f"archived recording mismatch: {changes}/{len(test)} decisions, max {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("phase", choices=("inputs", "reconstruct"))
    args = parser.parse_args()
    run(args.root, args.phase)
