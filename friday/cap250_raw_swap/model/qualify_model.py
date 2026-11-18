"""Authenticate F16e and test TorchScript with current windows in the real worker image."""
import argparse
import copy
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from encoder312 import fingerprint
from external_inputs import ORDERS, PEOPLE, VISITS, ancestry, cohort, deliveries
from held import frozen_member
from held_distill import raw_probability
from incremental_model import export_actor
from replay_f13 import origin_report, sha


HERE = Path(__file__).resolve().parent
F16E = HERE.parent / "ni26_friday_s1c_distill_external_20261118"
LOCK_SHA = "bc383cbc561223a803ef597300de5cde5869034dc35e464f0852462a2593ca94"
TOL = 2e-5
SRC_CHECKPOINTS = (
    "bcedeb0fb01b1735a834f9c8a7270d04f97fe085abce31a5c0bb05d21aa275f5",
    "82b9ae356bc5e09131f3b5db3ced8674f1691e1839031c78e020844df233781d",
)
RUNTIME = "2.14.1+cu130"
STATES = ("hist_n", "hist_s1", "hist_s2", "hist_mass", "hist_w1", "hist_w2", "hist_clipped")


def authenticated(ix, seed):
    lockpath = F16E / "before_labels.json"
    if sha(lockpath) != LOCK_SHA:
        raise ValueError("independent F16e decision lock differs")
    lock = json.loads(lockpath.read_text())
    checkpoint = F16E / "checkpoints" / f"hard_{seed}_all_source.pt"
    receipt = checkpoint.with_suffix(".json")
    identity = lock["source"][f"s{seed}_hard"]
    meta = json.loads(receipt.read_text())
    if ((sha(checkpoint), sha(receipt), meta["state"], meta["seed"], meta["kind"], meta["held"]) !=
            (SRC_CHECKPOINTS[seed], identity["receipt"], identity["state"], seed, "hard", "all_source") or
            meta["checkpoint"] != identity["checkpoint"] or meta["held_labels_opened"] or
            meta["q_gradients_nonzero"] or meta["epochs"] != 40 or len(meta["source_people"]) != 10 or
            any(not p.startswith("Dreyer2026Proteus/") for p in meta["source_people"])):
        raise ValueError("raw hard-CE checkpoint or source exclusion differs")
    path = F16E / "banks" / f"person_{ix}.npz"
    if sha(path) != lock["banks"][path.name]:
        raise ValueError("independent F16e archived person decisions differ")
    with np.load(path, allow_pickle=False) as z:
        index, decisions = z["index"], z[f"s{seed}_hard_bounded"]
    return checkpoint, meta, index, decisions


def immutable_tensors(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()
            if name.rsplit(".", 1)[-1] not in STATES}


def ensure_reset(module):
    if module.reset().item() != 0:
        raise ValueError("TorchScript reset counter did not clear")
    history = {name.rsplit(".", 1)[-1]: tensor for name, tensor in module.named_buffers()
               if name.rsplit(".", 1)[-1].startswith("hist_")}
    if set(history) != set(STATES) or any(torch.count_nonzero(v).item() for v in history.values()):
        raise ValueError("reset missed a session history tensor")


@torch.no_grad()
def batch_oracle(seed, i, x, delivery, rows):
    path, meta, archive_rows, archived = authenticated(i, seed)
    if not np.array_equal(archive_rows, rows):
        raise ValueError("canonical Scherer rows differ from F16e")
    model, layer = frozen_member(path, meta)
    original = fingerprint(model)
    layer.forward = MethodType(lambda self, values: normalize(self, values, []), layer)
    result = {}
    for key, ordered in delivery.items():
        batch = raw_probability(model, x[ordered] / meta["scale"])
        j = ORDERS.index(key[0])
        if not np.array_equal(batch.argmax(1), archived[j, np.searchsorted(rows, ordered)]):
            raise ValueError("resumed CPU batch oracle did not reproduce F16e decisions")
        result[key] = batch
    if fingerprint(model) != original:
        raise ValueError("resumed batch oracle altered trained parameters")
    return meta, result


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def qualified(path, x, delivery, reference, device):
    # CUDA profiling can choose different arithmetic for cold and warmed methods.
    with torch.jit.optimized_execution(False):
        return qualified_without_profiling(path, x, delivery, reference, device)


def qualified_without_profiling(path, x, delivery, reference, device):
    module = torch.jit.load(str(path), map_location=device).eval()
    ensure_reset(module)
    before = immutable_tensors(module)
    worst, first_max, windows = 0., 0., 0
    timings = []
    for key, rows in delivery.items():
        frames = torch.from_numpy(np.ascontiguousarray(x[rows])).unsqueeze(1).to(device)
        other = delivery[(key[0], VISITS[1 - VISITS.index(key[1])])]
        cross = torch.from_numpy(np.ascontiguousarray(x[other[:10]])).unsqueeze(1).to(device)
        ensure_reset(module)
        fresh = copy.deepcopy(module)
        torch.testing.assert_close(fresh(frames[0]), module(frames[0]), rtol=0, atol=0)
        ensure_reset(module)

        start = time.perf_counter()
        live = torch.stack([module(window) for window in frames])
        sync(device)
        elapsed = time.perf_counter() - start
        timings.append(elapsed / len(frames))
        if live.shape != (len(frames), 1, 3) or live.dtype != torch.float64:
            raise ValueError("artifact returned a noncanonical class vector")
        score = live.cpu().numpy()[:, 0]
        target = reference[key]
        if (not np.isfinite(score).all() or not np.allclose(score.sum(1), 1.) or
                not np.array_equal(score.argmax(1), target.argmax(1)) or np.abs(score - target).max() > TOL):
            raise ValueError(f"{device} one-window oracle mismatch at {key}: "
                             f"{np.abs(score - target).max():.3g}, "
                             f"{np.count_nonzero(score.argmax(1) != target.argmax(1))} decisions")
        first_max = max(first_max, float(np.abs(score[0] - target[0]).max()))
        worst = max(worst, float(np.abs(score - target).max()))
        ensure_reset(module)
        prefix = torch.stack([module(point) for point in frames[:31]])
        if not torch.equal(prefix, live[:31]):
            raise ValueError("artifact future rows changed the truncated prefix")
        ensure_reset(module)
        for point in cross:
            module(point)
        ensure_reset(module)
        if not torch.equal(prefix, torch.stack([module(point) for point in frames[:31]])):
            raise ValueError("TorchScript artifact carried state across sessions")
        windows += len(frames)

    ensure_reset(module)
    window_name = next(iter(delivery))
    warm = torch.from_numpy(np.ascontiguousarray(x[delivery[window_name][:32]])).unsqueeze(1).to(device)
    for frame in warm:
        module(frame)
    sync(device)
    dispatch, host = [], []
    for _ in range(3):
        ensure_reset(module)
        start = time.perf_counter()
        for frame in warm:
            module(frame)
        sync(device)
        dispatch.append((time.perf_counter() - start) * 1000 / len(warm))
        ensure_reset(module)
        start = time.perf_counter()
        for frame in warm:
            module(frame).cpu()
        sync(device)
        host.append((time.perf_counter() - start) * 1000 / len(warm))
    ensure_reset(module)
    after = immutable_tensors(module)
    if set(before) != set(after) or any(not torch.equal(v, after[k]) for k, v in before.items()):
        raise ValueError("frozen model parameters or constants changed during qualification")
    return dict(windows=windows, max_abs=worst, first_max_abs=first_max, prefix_checks=len(delivery),
                reset_checks=len(delivery), warm_ms=float(np.median(dispatch)), warm_host_ms=float(np.median(host)))


@torch.no_grad()
def run(person_index, mode):
    torch.set_num_threads(1)
    if torch.__version__ != RUNTIME or not torch.cuda.is_available():
        raise ValueError("requires exact CUDA worker runtime")
    origin_report()
    ancestry()
    x, subject, _, _, stream = cohort(waveforms=True)
    name = PEOPLE[person_index]
    rows = np.flatnonzero(subject == name)
    delivery = deliveries(stream, name)
    if mode != "export":
        receipt = json.loads((HERE / "model-receipt.json").read_text())
        if receipt["source"] != sha(HERE / "source.sha256"):
            raise ValueError("artifact source differs from smoke receipt")
    records = {}
    for seed in (0, 1):
        checkpoint, _, _, _ = authenticated(person_index, seed)
        meta, oracle = batch_oracle(seed, person_index, x, delivery, rows)
        path = HERE / f"raw_s1c_s{seed}.pt"
        if mode == "export":
            model, layer = frozen_member(checkpoint, meta)
            first = next(iter(delivery.values()))[0]
            compiled = export_actor(model, layer, meta["scale"], torch.from_numpy(x[first:first + 1]))
            with path.open("xb") as f:
                torch.jit.save(compiled, f)
        elif sha(path) != receipt["seeds"][str(seed)]["artifact"]:
            raise ValueError("artifact SHA differs from the real worker smoke")
        records[str(seed)] = dict(artifact=sha(path), checkpoint=sha(checkpoint), scale=meta["scale"],
                                  cpu=qualified(path, x, delivery, oracle, torch.device("cpu")),
                                  cuda=qualified(path, x, delivery, oracle, torch.device("cuda")))
        print(json.dumps(records[str(seed)], sort_keys=True), flush=True)
    report = dict(seeds=records, source=sha(HERE / "source.sha256"), person=name, orders=ORDERS,
                  device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  class_order=["motor imagery", "mental subtraction", "word generation"], jit_optimized_execution=False,
                  held_gold_opened=False)
    output = HERE / ("model-receipt.json" if mode == "export" else f"check-{person_index}.json")
    with output.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"PASS complete raw one-window qualification {sha(output)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("export", "complete"))
    p.add_argument("person", type=int, choices=range(len(PEOPLE)))
    a = p.parse_args()
    run(a.person, a.mode)
