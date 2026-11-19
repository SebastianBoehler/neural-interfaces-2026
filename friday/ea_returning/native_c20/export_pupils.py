"""Export and qualify the exact hard pupils on genuine returning deliveries."""

import argparse
import copy
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from encoder312 import fingerprint
from eye_scope import DATA, DATA_SHA, ORDER_FILE, ORDER_SHA, PEOPLE, pin, rows_sha, sha
from held import frozen_member
from held_distill import raw_probability
from incremental_model import export_actor
from pupil_source import source
from replay_f13 import origin_report


NAMES = ("hist_n", "hist_s1", "hist_s2", "hist_mass", "hist_w1", "hist_w2", "hist_clipped")
TOL = 2e-5


def reset(actor):
    if actor.reset().item() != 0:
        raise ValueError("original S1C actor reset failed")
    items = {name.rsplit(".", 1)[-1]: x for name, x in actor.named_buffers() if ".hist_" in name}
    if set(items) != set(NAMES) or any(torch.count_nonzero(v).item() for v in items.values()):
        raise ValueError("S1C retained session-level buffers")


def frozen(actor):
    return {name: x.detach().clone() for name, x in actor.state_dict().items() if name.rsplit(".", 1)[-1] not in NAMES}


@torch.no_grad()
def qualify(path, x, oracle, cross):
    with torch.jit.optimized_execution(False):
        actor = torch.jit.load(str(path), map_location="cpu").eval()
        reset(actor)
        before = frozen(actor)
        prefix_clone = copy.deepcopy(actor)
        frames = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1)
        context = torch.from_numpy(np.ascontiguousarray(cross)).unsqueeze(1)
        actual = torch.stack([actor(row) for row in frames]).numpy()[:, 0]
        reset(actor)
        prefix = torch.stack([actor(row) for row in frames[:17]]).numpy()[:, 0]
        np.testing.assert_array_equal(prefix, actual[:17])
        np.testing.assert_array_equal(prefix_clone(frames[0]).numpy()[0], prefix[0])
        reset(actor)
        for row in context:
            actor(row)
        reset(actor)
        np.testing.assert_array_equal(prefix, torch.stack([actor(row) for row in frames[:17]]).numpy()[:, 0])
        after = frozen(actor)
        if set(after) != set(before) or any(not torch.equal(v, after[k]) for k, v in before.items()):
            raise ValueError("same-state S1C actor parameters changed")
    error = float(np.abs(actual - oracle).max())
    if error > TOL or not np.isfinite(actual).all():
        raise ValueError(f"actual one-window F17 bounded returning oracle failed: {error}")
    np.testing.assert_array_equal(actual.argmax(1), oracle.argmax(1))
    return error


@torch.no_grad()
def export(who, seed, x, subj, visit, original, native):
    c, meta, rows, signed = source(who, seed, subj, visit)
    m, layer = frozen_member(c, meta)
    m.skip = False
    layer.forward = MethodType(lambda self, x: normalize(self, x, []), layer)
    old_scope, matched, proof = subj == f"Dreyer2026Proteus/{who}", {}, {}
    for day in ("1", "2", "3"):
        selected = old_scope & (visit == day)
        audit = original[selected[original]]
        a = raw_probability(m, x[audit] / meta["scale"])
        np.testing.assert_array_equal(a.argmax(1), signed[np.searchsorted(rows, audit)])
        if day == "1":
            continue
        seq = native[selected[native]]
        if len(seq) != len(audit) or np.array_equal(seq, audit):
            raise ValueError("returning oracle reused a different protocol's order")
        matched[day] = seq, raw_probability(m, x[seq] / meta["scale"])
    path = Path(who) / f"hard_s{seed}.pt"
    sample = torch.from_numpy(x[matched["2"][0][:1]].copy())
    actor = export_actor(m, layer, meta["scale"], sample)
    with path.open("xb") as f:
        torch.jit.save(actor, f)
    path.chmod(0o444)
    values = {}
    for day, (r, p) in matched.items():
        other = matched["3" if day == "2" else "2"][0][:9]
        proof[day] = dict(rows=len(r), delivery=rows_sha(r), max_abs=qualify(path, x[r], p, x[other]),
                          exact_oracle_classes=True, exact_prefix_reset=True, exact_first=True)
        values[f"rows{day}"], values[f"oracle{day}"] = r, p
    if fingerprint(m) != meta["state"]:
        raise ValueError("source fitted artifact changed during independent compilation")
    oracle = path.with_suffix(".npz")
    with oracle.open("xb") as f:
        np.savez_compressed(f, **values)
    oracle.chmod(0o444)
    p = dict(person=who, seed=seed, source=sha("source.sha256"), source_checkpoint=sha(c), state=meta["state"],
             fit_rows=meta["source_rows"], fit_people=meta["source_people"], order=ORDER_SHA, data=DATA_SHA,
             signed_all_loso_exact=True, own_visit1_excluded=True, held_labels_opened=False, visits=proof,
             artifact=sha(path), oracle=sha(oracle), runtime=str(torch.__version__))
    receipt = path.with_suffix(".json")
    with receipt.open("x") as f:
        json.dump(p, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    receipt.chmod(0o444)
    print("PASS exact returning pupils", who, seed, sha(receipt), proof, flush=True)


def main(who):
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("requires the pinned actual CPU runtime")
    torch.set_num_threads(1)
    origin_report()
    pin(DATA, DATA_SHA)
    pin(ORDER_FILE, ORDER_SHA)
    with np.load(DATA, allow_pickle=False) as f:
        x, subj, visit = f["X"], f["subject"], f["session"]
    with np.load(ORDER_FILE, allow_pickle=False) as z:
        original, native = z["full|0|all_loso"], z["full|0|seen_session"]
    for seed in (0, 1):
        export(who, seed, x, subj, visit, original, native)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("person", choices=PEOPLE)
    main(cli.parse_args().person)
