"""Recompute the three exact frozen F25 pupil arms independently."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from encoder312 import fingerprint
from held_modulation import frozen_member, paired
from pins import ACTOR, C20, C20_PUPILS, C20_PUPIL_SOURCE, C20_RECEIPTS, F25, F25_BLIND, F25_SCORER, F25_RESULTS, ORDERS, PUPIL_ARMS
from pupils import do_order
from replay_f13 import DATA_SHA, ORDER_SHA, ORDERS as ORIGINAL_ORDERS, origin_report, raw_inputs

HERE = Path(__file__).resolve().parent

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(person, seed, scope):
    torch.set_num_threads(2)
    origin_report()
    assert sha(F25/"before_labels.sha256") == F25_BLIND
    assert sha(F25/"score.sha256") == F25_SCORER and sha(F25/"whole_results.json") == F25_RESULTS
    assert sha(C20/"full0.json") == C20_RECEIPTS["full0"] and sha(C20_PUPILS/"source.sha256") == C20_PUPIL_SOURCE
    x, subject, people, stream, perms = raw_inputs()
    visit = np.array([s.rsplit("|",1)[-1] for s in stream])
    rows = np.flatnonzero(people == person)
    name = f"{person}_s{seed}.npz"
    path = F25/"folds"/name
    prior_path = C20_PUPILS/"pupils"/name
    c20_folds = json.loads((C20/"full0.json").read_text())["folds"]
    assert sha(prior_path) == c20_folds[name]
    assert sha(path) == path.with_suffix(".sha256").read_text().strip()
    checks = paired(person, seed, False, subject, visit)
    bank = {}
    with np.load(path, allow_pickle=False) as z, np.load(prior_path, allow_pickle=False) as previous:
        source = json.loads(str(z["metadata"]))
        np.testing.assert_array_equal(z["index"], rows)
        np.testing.assert_array_equal(previous["index"], rows)
        assert source["actor"] == ACTOR and source["data"] == DATA_SHA and source["order_sha"] == ORDER_SHA
        assert source["orders"] == list(ORIGINAL_ORDERS) and not source["held_labels_opened"]
        for arm in PUPIL_ARMS:
            model_path, info = checks[arm]
            proof = source["evidence"][arm]
            assert proof["checkpoint"] == info["checkpoint"] and proof["state"] == info["state"]
            assert proof["receipt"] == sha(model_path.with_suffix(".json"))
            assert not info["held_labels_opened"] and set(info["source_people"]) == set(subject[people != person])
            model, layer = frozen_member(model_path, info)
            model.skip = False
            for order in ORDERS:
                k = ORIGINAL_ORDERS.index(order)
                p, diff, active = do_order(model, layer, x, stream, rows, perms[order], info["scale"])
                np.testing.assert_array_equal(p.argmax(1), z[arm+"_bounded"][k])
                if arm == "hard":
                    np.testing.assert_array_equal(p, previous[f"hard|{order}"])
                bank[f"{arm}|{order}"] = p
                bank[f"{arm}|{order}|proof"] = np.array([diff, active])
            assert info["state"] == fingerprint(model)
    destination = HERE/scope/f"{person}_s{seed}.npz"
    assert len(rows) and all(isinstance(x,np.ndarray) for x in bank.values())
    evidence = dict(actor=ACTOR, fold=sha(path), c20_hard=sha(prior_path), source=sha(HERE/"source.sha256"),
                    data=DATA_SHA, order_sha=ORDER_SHA,
                    person=person, seed=seed, held_labels_opened=False,
                    states={k:checks[k][1]["state"] for k in PUPIL_ARMS},
                    checkpoints={k:checks[k][1]["checkpoint"] for k in PUPIL_ARMS})
    with destination.open("xb") as f:
        np.savez_compressed(f, index=rows, metadata=json.dumps(evidence,sort_keys=True), **bank)
    destination.chmod(0o444)
    print("PASS F25c pupils",person,seed,scope,sha(destination),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("person")
    p.add_argument("seed", type=int, choices=(0,1))
    p.add_argument("scope", choices=("smoke", "pupils"))
    a=p.parse_args()
    run(a.person,a.seed,a.scope)
