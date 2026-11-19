"""Consume independent A100 pupils in the original corrected cap250 native tails."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha
from complete_zero import FEAT, HASHES
from outer_f25 import arms
from pins import ARMS, C20, C20_ARRAYS, C20_RECEIPTS, ORDERS, PUPIL_ARMS
from worker import run_tail


WORK = Path(__file__).resolve().parent
F33A = WORK.parent / "ni26_friday_s1c_covariance_a100_whole_20261119"
F33A_BLIND = "f996e3a55d376f6856bbd07425632033924a175636acc21b332d1a4d3f4f0525"
F25_ACTOR = "589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c"
ORDERS_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"


def read_pupils(order, people, visit, pupil_sha, actor_sha):
    pin(F33A / "before_labels.sha256", F33A_BLIND)
    independent = dict(line.split()[::-1] for line in (F33A / "before_labels.sha256").read_text().splitlines())
    frames, checks = {}, {}
    bank = {k:np.full((2, len(people), 3), np.nan) for k in PUPIL_ARMS}
    clips = dict.fromkeys(PUPIL_ARMS, 0)
    for person in sorted(set(people)):
        rows = np.flatnonzero(people == person)
        expected_visits = 2 * len(set(visit[rows]))
        for seed in (0,1):
            key = f"{person}_s{seed}.npz"
            path = WORK / "pupils" / key
            seal = path.with_suffix(".sha256")
            if path.stat().st_mode & 0o222 or seal.stat().st_mode & 0o222:
                raise ValueError("held A100 pupil was not sealed as read-only")
            digest = seal.read_text().strip()
            pin(path, digest)
            frames[key] = digest
            with np.load(path, allow_pickle=False) as fold:
                ev = json.loads(str(fold["metadata"]))
                expected = dict(person=person, seed=seed, data=HASHES[FEAT], order_sha=ORDERS_SHA,
                                f25=F25_ACTOR, device="NVIDIA A100-PCIE-40GB", weights=actor_sha, source=pupil_sha,
                                sealed_hard_fold=independent[f"folds/{key}"], bare_equals_hard=True,
                                held_labels_opened=False)
                if any(ev[k] != v for k,v in expected.items()) or set(ev["arms"]) != set(PUPIL_ARMS):
                    raise ValueError("held A100 source exclusion, order or hard reference proof differs")
                np.testing.assert_array_equal(fold["index"], rows)
                checks[key] = {}
                for name, dest in bank.items():
                    provenance = ev["arms"][name]
                    checkpoint = WORK / "checkpoints" / f"{name}_{seed}_{person}.pt"
                    pin(checkpoint.with_suffix(".json"), provenance["receipt"])
                    pin(checkpoint, provenance["checkpoint"])
                    if (provenance["exact_first_windows"] != expected_visits
                            or provenance["exact_interleaved_resets"] != expected_visits):
                        raise ValueError("held source actor causal proof incomplete")
                    p, proof = fold[f"{name}|{order}"], fold[f"{name}|{order}|proof"]
                    if (p.shape != (len(rows),3) or proof.shape != (2,) or not 0 <= proof[0] <= 2e-5
                            or not proof[1] >= 0 or not np.isfinite(proof).all() or not np.isnan(dest[seed,rows]).all()):
                        raise ValueError("held A100 probabilities or observed causal prefix invalid")
                    clips[name] += int(proof[1])
                    dest[seed, rows] = p
                    checks[key][name] = provenance["receipt"]
    if len(frames) != 20 or not all(clips.values()):
        raise ValueError("complete 20-fit bound-active bank is required")
    for values in bank.values():
        assert np.isfinite(values).all() and (values >= 0).all()
        np.testing.assert_allclose(values.sum(2), 1., rtol=0, atol=1e-12)
    if any(np.array_equal(bank[a],bank[b]) for a,b in (("single","hard"),("multi","hard"),("single","multi"))):
        raise ValueError("distinct trained pupil families produced a duplicate complete posterior bank")
    return dict(identity=None, **{arm:p.mean(0) for arm,p in bank.items()}), frames, checks, clips


def run(order, smoke, manifests):
    started = perf_counter()
    src = {name:sha(WORK / f"{name}.sha256") for name in manifests}
    if not smoke:
        witness = json.loads((WORK / "whole-smoke.json").read_text())
        assert witness["passed"] and witness["source"] == src and not witness["held_labels_opened"]
    pin(FEAT, HASHES[FEAT])
    with np.load(FEAT, allow_pickle=False) as data:
        people = np.array([name.split("/")[-1] for name in data["subject"]])
        visits = data["session"]
    assert len(people) == 2655 and len(set(people)) == 10
    pupils, folds, checks, clips = read_pupils(order, people, visits, src["pupil"], src["actor"])
    if not smoke:
        assert witness["folds"] == folds and witness["fit_receipts"] == checks
    active = people == "P01" if smoke else np.ones(len(people), dtype=bool)
    pin(C20 / f"{order}.json", C20_RECEIPTS[order])
    pin(C20 / f"{order}.npz", C20_ARRAYS[order])
    with arms(order, pupils, active) as ctx, np.load(C20 / f"{order}.npz", allow_pickle=False) as prior:
        count = ctx["count"]
        sessions = tuple(count.SESS)
        rows = {s:count.C.order_rows(order,s,count.PROTO) for s in sessions}
        assert len(sessions) == 14
        p01 = next(s for s,r in rows.items() if set(people[r])=={"P01"} and set(visits[r])=={"1"})
        latches = count.latch(order)
        peer = next(s for s in sessions if s!=p01 and latches[s] is not None and latches[s]<120)
        prefixes = (p01,peer,p01)
        selected = (p01,peer) if smoke else sessions
        index = np.concatenate([rows[s] for s in selected])
        np.testing.assert_array_equal(prior["delivery"],ctx["delivery"])
        if not smoke:
            np.testing.assert_array_equal(index,ctx["delivery"])
        paths, work = {}, {}
        for key in ARMS:
            paths[key], work[key] = run_tail(ctx,order,ctx["raw"][key],selected,prefixes)
        for stage, stock in paths["stock"].items():
            np.testing.assert_array_equal(stock[index],ctx["reference"][stage][index])
            np.testing.assert_array_equal(stock[index],prior["stock|"+stage][index])
            np.testing.assert_array_equal(stock[index],paths["identity"][stage][index])
            if smoke:
                for name in PUPIL_ARMS:
                    np.testing.assert_array_equal(stock[rows[peer]],paths[name][stage][rows[peer]])
        assert all(v["count_effect"] > 0 for v in work.values())
        if not smoke:
            assert all((v["csp_refits"],v["csp_fusions"]) == (126,1395) for v in work.values())
        movement = {}
        for name in PUPIL_ARMS:
            movement[name] = float(np.abs(paths[name]["terminal"][index] - paths["stock"]["terminal"][index]).max())
            assert movement[name] > 1e-4
            assert any(work[name]["csp"][str(s)]["target_sha"] != work["stock"]["csp"][str(s)]["target_sha"]
                       for s in selected)
        assert all(np.abs(paths[a]["terminal"][index] - paths[b]["terminal"][index]).max() > 1e-4
                   for a,b in (("single","hard"),("multi","hard"),("single","multi")))
        report = dict(passed=True, scope="smoke" if smoke else "complete", order=order, source=src, parent=ctx["parent"],
                      archive_zero=ctx["zero"], rows=len(index), sessions=len(selected), work=work,
                      folds=folds, fit_receipts=checks, bounded_clips=clips, held_labels_opened=False,
                      prefixes=list(map(str,prefixes)), stock_bitwise=True, native_identity=True,
                      actual_warm_delta=True, original_shadow=True, new_csp_targets=True,
                      outer=ctx["proof"], movement=movement, seconds=perf_counter()-started)
        if not smoke:
            bank = dict(index=np.arange(len(index)), delivery=index,
                        **{f"{arm}|{k}":v for arm in ARMS for k,v in paths[arm].items()})
            report["output"] = write_npz(WORK / f"{order}.npz",bank)
        destination = WORK / ("whole-smoke.json" if smoke else f"{order}.json")
        write_receipt(destination,**report)
    print("PASS F-C22h", order, report["scope"], sha(destination), movement, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("order", choices=ORDERS)
    cli.add_argument("mode", choices=("smoke","complete"))
    a = cli.parse_args()
    assert a.mode != "smoke" or a.order == "full0"
    run(a.order, a.mode == "smoke", ("actor","pupil","whole"))
