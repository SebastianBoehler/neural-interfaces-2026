"""Test the historical r15c250 implementation on one real held-new session."""

import hashlib
import json
import os
from pathlib import Path
import runpy

import numpy as np
import torch

from source_labels import read_training_labels


ROOT = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
ARCHIVE = ROOT.parent / "ni26_s318_coral_fbl_20261117/runs/diag/ttt/probs18"
HERE = Path(__file__).resolve().parent
HELD = "Dreyer2026Proteus/P01"
TOL = 2e-5


def file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_data(held=HELD):
    path = ROOT / "data/proteus_windows.npz"
    with np.load(path, allow_pickle=False) as store:
        data = {name: store[name] for name in store.files if name != "y"}
    subject = data["subject"]
    if data["X"].shape != (2655, 41, 480) or data["X"].dtype != np.float32:
        raise ValueError("unexpected official waveform input")
    fit = np.flatnonzero(subject != held)
    test = np.flatnonzero(subject == held)
    if not len(test) or len(set(subject[fit])) != 9:
        raise ValueError("archival smoke fold differs")
    data["y"] = np.full(len(subject), -1, dtype=np.int64)
    data["y"][fit] = read_training_labels(path, fit, len(subject))
    data["stream"] = np.char.add(np.char.add(subject, "|"), data["session"])
    data["w"] = np.ones(len(subject))
    data["nolow"] = np.zeros(len(subject), dtype=bool)
    data["tag"] = ""
    return data, fit, test


def changed(actual, reference):
    if not np.isfinite(actual).all() or actual.shape != reference.shape:
        raise ValueError("invalid smoke probabilities")
    return dict(max_abs=float(np.max(np.abs(actual - reference))),
                changed=int(np.count_nonzero(actual.argmax(1) != reference.argmax(1))))


def refit_api(device_name="A100"):
    if not torch.cuda.is_available() or device_name not in torch.cuda.get_device_name():
        raise ValueError(f"historical test requires {device_name}")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    os.environ.update(NI26_WS=str(ROOT), SCHEME="full", PSEED="0", KIND="v51b", ARMS="s15,r15",
                      CAPS="250", DRAW="0", CHAIN_NPZ=str(ROOT / "runs/labels/chain_v51b_full0.npz"),
                      RELAB_NPZ=str(ROOT / "runs/labels/relab_v51b_full0.npz"))
    source = ROOT / "studies/14nov/ftrelabel/ftrel.py"
    scope = runpy.run_path(str(source))
    return scope["T5"], scope["stream"], file_sha(source)


def prepare():
    t5, stream, source_sha = refit_api()
    data, fit, test = source_data()
    if len(fit) != 2465 or len(test) != 190 or set(data["session"][test]) != {"1"}:
        raise ValueError("archival smoke fold differs")
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as ranked:
        order = ranked["full|0|all_loso"]
    rows = order[data["subject"][order] == HELD]
    np.testing.assert_array_equal(np.sort(rows), test)
    pseudo = t5.chain_labels("all_loso")
    target = np.stack([pseudo[int(i)] for i in rows])
    np.testing.assert_array_equal(target[:, -1], np.arange(len(rows)))
    model = t5.make("fbl", data, 0)
    model.fit(fit)
    model.set_k(t5.ARCHS["fbl"][1])
    if model.dev != "cuda" or model.n_sbn < 1 or not (data["y"][test] == -1).all():
        raise ValueError("wrong FBLight or source exclusion")
    x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
    with torch.no_grad():
        frozen = torch.softmax(model.net(x)[:, :3].double(), 1).cpu().numpy()
    expected = {}
    for name in ("base", "r15c250"):
        with np.load(ARCHIVE / f"ttt18full0LPC_fbl_{name}-all_loso-s0.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["index"], np.arange(len(data["subject"])))
            expected[name] = archive["proba"][rows]
    report = dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), held=HELD,
                  held_labels_opened=False, source_sha=source_sha, fit_rows=int(len(fit)),
                  order_rows=int(len(rows)), frozen=changed(frozen, expected["base"]))
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["frozen"]["changed"] or report["frozen"]["max_abs"] > TOL:
        raise ValueError("source fit failed archived frozen parity before refitting")
    return model, x, rows, target, frozen, expected["r15c250"], stream, report


def main():
    model, x, _, target, frozen, expected, stream, report = prepare()
    replay = stream(model, x, target, 0)
    np.testing.assert_array_equal(replay["base"], frozen)
    np.testing.assert_array_equal(replay["r15c250"][:20], frozen[:20])
    report["r15c250"] = changed(replay["r15c250"], expected)
    report["refit_activity"] = changed(replay["r15c250"], frozen)
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["r15c250"]["changed"] or report["r15c250"]["max_abs"] > TOL or not report["refit_activity"]["changed"]:
        raise ValueError("refit failed archived parity or activity")
    with (HERE / "fbl_seed0_p01_full0_receipt.json").open("x") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
