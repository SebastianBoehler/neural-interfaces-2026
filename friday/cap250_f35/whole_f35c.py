"""Run matched F35 branches through the proved corrected cap250 pipeline."""
import argparse
from itertools import combinations
import json

import numpy as np

from bridge_owner import pin, sha
from pins_f35c import ACTOR, BLIND, DATA, DELIVERY, DEVICE, F35, ORDERS, PUPIL_ARMS, REF_BANKS, REF_BLIND, REFERENCE, WORK
from whole_f25a import run


def read_pupils(order, people, visit, source):
    pin(F35/"before_labels.sha256", BLIND)
    pin(REFERENCE/"before-labels.json", REF_BLIND)
    f35 = dict(line.split()[::-1] for line in (F35/"before_labels.sha256").read_text().splitlines())
    reference = json.loads((REFERENCE/"before-labels.json").read_text())["orders"][order]["folds"]
    bank = {arm:np.full((2,len(people),3),np.nan) for arm in PUPIL_ARMS}
    clips = dict.fromkeys(PUPIL_ARMS,0)
    frames, checks = {}, {}
    for person in sorted(set(people)):
        rows = np.flatnonzero(people==person)
        expected_visits = 2*len(set(visit[rows]))
        for seed in (0,1):
            key = f"{person}_s{seed}.npz"
            path = WORK/"pupils"/key
            seal = path.with_suffix(".sha256")
            if (path.stat().st_mode | seal.stat().st_mode) & 0o222:
                raise ValueError("authenticated F35 bounded pupil is writable")
            frames[key] = digest = seal.read_text().strip()
            pin(path,digest)
            checks[key] = {}
            with np.load(path,allow_pickle=False) as fold:
                ev = json.loads(str(fold["metadata"]))
                want = dict(person=person,seed=seed,data=DATA,order_sha=DELIVERY,actor=ACTOR,device=DEVICE,
                            source=source["pupil"],signed_f35=f35[f"folds/{key}"],hard_reference=reference[key],
                            hard_bitwise=True,held_labels_opened=False)
                if any(ev[k] != v for k,v in want.items()) or set(ev["arms"]) != set(PUPIL_ARMS):
                    raise ValueError("held actor signed source, reference or scope differs")
                np.testing.assert_array_equal(fold["index"],rows)
                for arm,dest in bank.items():
                    proof = ev["arms"][arm]
                    fit = F35/"checkpoints"/f"{arm}_{seed}_{person}.pt"
                    pin(fit.with_suffix(".json"),proof["receipt"])
                    pin(fit,proof["checkpoint"])
                    if (proof["exact_first_windows"] != expected_visits
                            or proof["exact_interleaved_resets"] != expected_visits):
                        raise ValueError("causal source session checks incomplete")
                    p, d = fold[f"{arm}|{order}"], fold[f"{arm}|{order}|proof"]
                    if (p.shape != (len(rows),3) or d.shape != (2,) or not np.isfinite(d).all()
                            or not 0 <= d[0] <= 2e-5 or d[1] < 0 or not np.isnan(dest[seed,rows]).all()):
                        raise ValueError("invalid bounded held F35 rows or causal prefix")
                    clips[arm] += int(d[1])
                    checks[key][arm] = proof["receipt"]
                    dest[seed,rows] = p
    assert len(frames) == 20 and all(clips.values())
    for p in bank.values():
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(2),1.,atol=1e-12,rtol=0)
    if any(np.array_equal(bank[a],bank[b]) for a,b in combinations(PUPIL_ARMS,2)):
        raise ValueError("duplicate fitted complete bounded arrays")
    return {arm:p.mean(0) for arm,p in bank.items()},frames,checks,clips


def exact_hard(order, paths, rows):
    path = REFERENCE/f"{order}.npz"
    pin(path,REF_BANKS[order])
    with np.load(path,allow_pickle=False) as original:
        for arm in ("stock","hard"):
            for stage,v in paths[arm].items():
                np.testing.assert_array_equal(v[rows],original[f"{arm}|{stage}"][rows])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("order",choices=ORDERS)
    cli.add_argument("mode",choices=("smoke","complete"))
    a = cli.parse_args()
    assert a.mode != "smoke" or a.order == "full0"
    run(a.order,a.mode == "smoke",("pupil","whole"),workspace=WORK,reader=read_pupils,
        arm_names=PUPIL_ARMS,hard_reference=exact_hard)
