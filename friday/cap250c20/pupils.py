"""Recover probabilities from the two authenticated source-fold F17 pupils."""

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from encoder312 import fingerprint
from held import frozen_member
from held_distill import raw_probability
from held_phase import ACTOR, paired
from replay_f13 import DATA_SHA, ORDER_SHA, ORDERS as ORIGINAL_ORDERS, origin_report, raw_inputs

HERE = Path(__file__).resolve().parent
F17 = Path("/mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_phase_20261118")
F17_BLIND = "77a25afbba7061dde2c781f3e8fb810b49d653aa57d09dcbc4b5abc0bb2abb37"
ORDERS = ("full0", "within0")
ARMS = ("hard", "jitter")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@torch.no_grad()
def do_order(model, layer, raw, stream, rows, order, scale):
    t = np.full((len(rows), 3), np.nan)
    maximum, active = 0., 0
    counts = []
    forward = layer.forward
    layer.forward = MethodType(lambda this, z: normalize(this, z, counts), layer)
    try:
        for visit in dict.fromkeys(stream[rows]):
            visit_rows = order[stream[order] == visit]
            other = order[stream[order] != visit][:16]
            x = raw[visit_rows] / scale
            counts.clear()
            value = raw_probability(model, x)
            assert len(counts) == 1
            active += counts[0]
            p = raw_probability(model, x[:16])
            raw_probability(model, raw[other] / scale)
            np.testing.assert_array_equal(p, raw_probability(model, x[:16]))
            np.testing.assert_array_equal(value[:16].argmax(1), p.argmax(1))
            maximum = max(maximum, float(np.abs(value[:16]-p).max()))
            assert maximum <= 2e-5
            t[np.searchsorted(rows, visit_rows)] = value
    finally:
        layer.forward = forward
    np.testing.assert_allclose(t.sum(1), 1., atol=1e-12, rtol=0)
    assert np.isfinite(t).all() and (t >= 0).all()
    return t, float(maximum), int(active)


def run(person, seed, scope):
    torch.set_num_threads(2)
    origin_report()
    assert sha(F17/"before_labels.sha256") == F17_BLIND
    x, subject, people, stream, perms = raw_inputs()
    visit = np.array([s.rsplit("|",1)[-1] for s in stream])
    rows = np.flatnonzero(people == person)
    path = F17/"folds"/f"{person}_s{seed}.npz"
    assert sha(path) == path.with_suffix(".sha256").read_text().strip()
    checks = paired(person, seed, False, subject, visit)
    bank = {}
    with np.load(path, allow_pickle=False) as z:
        source = json.loads(str(z["metadata"]))
        np.testing.assert_array_equal(z["index"], rows)
        assert source["actor"] == ACTOR and source["data"] == DATA_SHA and source["order_sha"] == ORDER_SHA
        assert source["orders"] == list(ORIGINAL_ORDERS) and not source["held_labels_opened"]
        for arm in ARMS:
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
                bank[f"{arm}|{order}"] = p
                bank[f"{arm}|{order}|proof"] = np.array([diff, active])
            assert info["state"] == fingerprint(model)
    destination = HERE/scope/f"{person}_s{seed}.npz"
    assert len(rows) and all(isinstance(x,np.ndarray) for x in bank.values())
    evidence = dict(actor=ACTOR, fold=sha(path), source=sha(HERE/"source.sha256"), data=DATA_SHA, order_sha=ORDER_SHA,
                    person=person, seed=seed, held_labels_opened=False,
                    states={k:checks[k][1]["state"] for k in ARMS},
                    checkpoints={k:checks[k][1]["checkpoint"] for k in ARMS})
    with destination.open("xb") as f:
        np.savez_compressed(f, index=rows, metadata=json.dumps(evidence,sort_keys=True), **bank)
    destination.chmod(0o444)
    print("PASS F-C20 pupils",person,seed,scope,sha(destination),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("person")
    p.add_argument("seed", type=int, choices=(0,1))
    p.add_argument("scope", choices=("smoke", "pupils"))
    a=p.parse_args()
    run(a.person,a.seed,a.scope)
