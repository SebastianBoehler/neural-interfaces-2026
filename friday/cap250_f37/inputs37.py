"""Reauthenticate all F37 posteriors and both hard anchors before native replay."""
from itertools import combinations
import json

import numpy as np

from bridge_owner import pin, sha
from pins_f37c import ACTOR, C20_PUPILS, DATA, DELIVERY, DEVICE, F32C, F37, PUPILS, WORK, read_folds


def read_pupils(order, people, visit, source):
    signed, reference, f32 = read_folds(sha)
    bank = {arm:np.full((2,len(people),3),np.nan) for arm in PUPILS}
    clips = dict.fromkeys(PUPILS,0)
    frames, checks = {}, {}
    for person in sorted(set(people)):
        rows = np.flatnonzero(people==person)
        expected_visits = 2*len(set(visit[rows]))
        for seed in (0,1):
            key = f"{person}_s{seed}.npz"
            path = WORK/"pupils"/key
            seal = path.with_suffix(".sha256")
            if (path.stat().st_mode | seal.stat().st_mode) & 0o222:
                raise ValueError("authenticated F37 bounded pupil is writable")
            frames[key] = digest = seal.read_text().strip()
            pin(path,digest)
            checks[key] = {}
            with (np.load(path,allow_pickle=False) as fold,
                  np.load(C20_PUPILS/"pupils"/key,allow_pickle=False) as hard,
                  np.load(F32C/"pupils"/key,allow_pickle=False) as matched):
                ev = json.loads(str(fold["metadata"]))
                want = dict(person=person,seed=seed,data=DATA,order_sha=DELIVERY,actor=ACTOR,device=DEVICE,
                            source=source,signed_f37=signed[key],hard_reference=reference[key],f32_reference=f32[key],
                            hard_bitwise=True,held_labels_opened=False)
                if any(ev[k] != v for k,v in want.items()) or set(ev["arms"]) != set(PUPILS):
                    raise ValueError("held actor signed source, reference or scope differs")
                for z in (fold,hard,matched):
                    np.testing.assert_array_equal(z["index"],rows)
                pin(C20_PUPILS/"pupils"/key,reference[key])
                pin(F32C/"pupils"/key,f32[key])
                pin(F37/"folds"/key,signed[key])
                for arm,dest in bank.items():
                    proof = ev["arms"][arm]
                    fit = F37/"checkpoints"/f"{arm}_{seed}_{person}.pt"
                    pin(fit.with_suffix(".json"),proof["receipt"])
                    pin(fit,proof["checkpoint"])
                    fit_proof = json.loads(fit.with_suffix(".json").read_text())
                    if fit_proof["tape"] != proof["tape"]:
                        raise ValueError("source perturbation tape changed")
                    if (proof["exact_first_windows"] != expected_visits
                            or proof["exact_interleaved_resets"] != expected_visits):
                        raise ValueError("causal source session checks incomplete")
                    p, d = fold[f"{arm}|{order}"], fold[f"{arm}|{order}|proof"]
                    if (p.shape != (len(rows),3) or d.shape != (2,) or not np.isfinite(d).all()
                            or not 0 <= d[0] <= 2e-5 or d[1] < 0 or not np.isnan(dest[seed,rows]).all()):
                        raise ValueError("invalid bounded held F37 rows or causal prefix")
                    if arm=="hard":
                        np.testing.assert_array_equal(p,hard[f"hard|{order}"])
                        np.testing.assert_array_equal(p,matched[f"hard|{order}"])
                    clips[arm] += int(d[1])
                    checks[key][arm] = proof["receipt"]
                    dest[seed,rows] = p
    assert len(frames) == 20 and all(clips.values())
    for p in bank.values():
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(2),1.,atol=1e-12,rtol=0)
    if any(np.array_equal(bank[a],bank[b]) for a,b in combinations(PUPILS,2)):
        raise ValueError("duplicate fitted complete bounded arrays")
    return dict(identity=None,**{arm:p.mean(0) for arm,p in bank.items()}),frames,checks,clips

