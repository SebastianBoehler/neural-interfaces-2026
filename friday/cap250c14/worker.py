"""Run full pinned cap250 sessions for the raw odds × directional EA factorial."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from history import ALIGNED, install as aligned_install
from inputs import ACTOR, ARMS, SEEDS, archive_parity, schedule, seal, sha
from odds import IDS, install as odds_install


def run(template, eeg, rows, arm, actor, *, trace=False):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    if arm in ("identity", "directional", "combined"):
        ea = aligned_install(model, "zero" if arm == "identity" else "directional")
    else:
        ea = None
    odds_on = arm in ("cognitive", "combined")
    odds = odds_install(model, actor if odds_on else None) if odds_on or arm == "identity" else None
    labels, aligned, members, actors = [], [], [], []
    aligned_hash = hashlib.sha256()
    exact_warm = 0
    for age, row in enumerate(rows):
        window = eeg[int(row):int(row) + 1].to("cuda:0")
        pred = model.predict(window)
        assert pred.shape == (1,) and pred.dtype == torch.int64 and pred.device.type == "cpu"
        assert not model.dead and not model.cp.dead and not model.cf.dead
        label = int(pred[0])
        assert 0 <= label <= 2
        labels.append(label)
        x = model.nets.last_inputs[ALIGNED[0]]
        assert x.shape == (41, 480) and all(model.nets.last_inputs[j] is x for j in ALIGNED)
        aligned_hash.update(np.ascontiguousarray(x))
        if odds is not None:
            got = odds.check(model.trace, age)
            exact_warm += int(odds_on and "fbraw" in model.trace["v6"]["p_ft"])
        if trace:
            aligned.append(x.copy())
            if odds is None:
                got = np.stack([model.trace["v6"]["members"][i] for i in IDS])
            members.append(got.copy())
            if odds_on:
                actors.append(odds.actor_q.copy())
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    activity = replay.activity(model)
    assert activity["windows"] == len(rows) and activity["errors"] == activity["ft_over"] == 0
    assert activity["ft_steps"] == 20 * activity["ft_refits"]
    assert odds is None or odds.calls == len(rows) and odds.late == exact_warm
    history = ea.receipt() if ea is not None else None
    if history is not None:
        assert history["windows"] == len(rows)
    info = dict(labels=labels, activity=activity, seconds=seconds, odds=odds.receipt() if odds else None,
                history=history, aligned_sha=aligned_hash.hexdigest())
    return info, (np.asarray(aligned), np.asarray(members), np.asarray(actors)) if trace else None


def posterior_parity(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error = float(np.abs(a-b).max())
    assert error <= 2e-5 and np.array_equal(a.argmax(axis=-1), b.argmax(axis=-1)), error
    return error


def exact_prefix(full, prefix):
    result, (x, p, a) = full
    short, (y, q, b) = prefix
    n = len(short["labels"])
    assert short["labels"] == result["labels"][:n] and np.array_equal(x[:n], y)
    assert np.array_equal(a[:n], b)
    return posterior_parity(p[:n], q)


def exposure(direction, combined):
    assert direction["aligned_sha"] == combined["aligned_sha"]
    for k in ("prior", "active_ages", "retained_traces"):
        assert direction["history"][k] == combined["history"][k], k


def smoke(template, actor, eeg, rows, other, archive):
    for arm in ("stock", "identity"):
        reference = run(template, eeg, rows, arm, actor, trace=True)
        archive_parity(reference[0], archive)
        if arm == "identity":
            identity = reference
        else:
            stock = reference
    duplicate = run(template, eeg, rows, "stock", actor, trace=True)
    archive_parity(duplicate[0], archive)
    errors = dict(identity=posterior_parity(stock[1][1], identity[1][1]),
                  stock=posterior_parity(stock[1][1], duplicate[1][1]))
    assert np.array_equal(stock[1][0], identity[1][0])
    assert identity[0]["odds"]["warm_calls"] == identity[0]["odds"]["max_pre"] == 0
    assert not identity[0]["history"]["active_ages"]
    print(f"PASS original exact control {errors}", flush=True)
    results = {"stock": stock[0], "identity": identity[0]}
    for arm in ARMS[1:]:
        full = run(template, eeg, rows, arm, actor, trace=True)
        run(template, eeg, other[:31], arm, actor)
        short = run(template, eeg, rows[:31], arm, actor, trace=True)
        err = exact_prefix(full, short)
        data, (aligned, _, _) = full
        assert data["activity"]["ft_refits"] > 0
        if data["history"] is not None:
            active = data["history"]["active_ages"]
            changed = int(np.any(aligned != stock[1][0], axis=(1, 2)).sum())
            assert sum(a < 119 for a in active) >= 2 and changed >= 6
            assert np.array_equal(aligned[:active[0] + 1], stock[1][0][:active[0] + 1])
            data["aligned_changed"] = changed
        else:
            assert data["aligned_sha"] == stock[0]["aligned_sha"]
        if data["odds"] is not None:
            assert data["odds"]["above_001"] >= 12 and data["odds"]["max_warm"] > 1e-5
        results[arm] = {**data, "prefix_posterior_max": err}
        print(f"PASS smoke {arm}: {data['activity']}, {data['odds']}", flush=True)
    exposure(results["directional"], results["combined"])
    order_rows = np.sort(rows)[:31]
    a = run(template, eeg, order_rows, "stock", actor, trace=True)
    b = run(template, eeg, order_rows, "identity", actor, trace=True)
    errors["recording_identity"] = exact_prefix(a, b)
    return dict(arms=results, stock_control_errors=errors, passed=True, rows=rows.tolist())


def produce(index, template, actor, eeg, who, deliveries, archive):
    names = [f"{who}|{visit}" for visit in (1, 2)]
    warm = deliveries[411][names[0]]
    anchors = {}
    for arm in ARMS:
        result, _ = run(template, eeg, warm, arm, actor)
        if arm == "stock":
            key = f"{names[0]}|411"
            archive_parity(result, archive[key])
            anchors[key] = dict(rows=warm.tolist(), run=result)
    second = deliveries[1411][names[0]]
    result, _ = run(template, eeg, second, "stock", actor)
    key = f"{names[0]}|1411"
    archive_parity(result, archive[key])
    anchors[key] = dict(rows=second.tolist(), run=result)
    out = {}
    for seed_index, seed in enumerate(SEEDS):
        for visit_index, name in enumerate(names):
            rows = deliveries[seed][name]
            rot = (index + 2 * seed_index + visit_index) % len(ARMS)
            seq = ARMS[rot:] + ARMS[:rot]
            results = {arm: run(template, eeg, rows, arm, actor)[0] for arm in seq}
            exposure(results["directional"], results["combined"])
            assert results["cognitive"]["aligned_sha"] == results["stock"]["aligned_sha"]
            assert all(results[arm]["odds"]["above_001"] > 0 for arm in ("cognitive", "combined"))
            key = f"{name}|{seed}"
            out[key] = dict(rows=rows.tolist(), order=seq, arms=results)
            print(f"SEALED {key}: " + " ".join(f"{a} {results[a]['seconds']:.2f}s" for a in seq), flush=True)
    return dict(sessions=out, archived_controls=anchors)


def main(mode, index):
    assert torch.__version__ == "2.14.1+cu130" and "A100" in torch.cuda.get_device_name()
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    assert sha("/actor.pt") == ACTOR
    with np.load("/external.npz", allow_pickle=False) as f:
        eeg = torch.from_numpy(f["X"])
        people, visits, channels = (f[k] for k in ("subject", "session", "ch_names"))
    assert eeg.shape == (2130, 41, 480) and eeg.dtype == torch.float32 and torch.isfinite(eeg).all()
    who = sorted(set(people))[index]
    deliveries = schedule(people, visits)
    prior = Path(f"/prior/cap_{index}.json")
    archive = json.loads(prior.read_text())
    for seed in (411, 1411):
        for day in (1, 2):
            name = f"{who}|{day}"
            assert deliveries[seed][name].tolist() == archive[f"{name}|{seed}"]["rows"]
    template = replay.load(Path("/package"), channels.tolist())
    assert template.perm is None and template.routed and template.ft.on and template.ft.cap == 250
    actor = torch.jit.load("/actor.pt", map_location="cuda:0").eval()
    if mode == "smoke":
        assert index == 5 and who == "Scherer2015/G"
        rows = deliveries[411][who+"|1"]
        assert len(rows) == 120
        other = deliveries[411][who+"|2"]
        payload = smoke(template, actor, eeg, rows, other, archive[who+"|1|411"])
        dest = Path("smoke.json")
        smoke_sha = None
    else:
        smoke_sha = sha("smoke.json")
        assert Path("smoke.sha256").read_text().strip() == smoke_sha
        assert json.loads(Path("smoke.json").read_text())["passed"]
        payload = produce(index, template, actor, eeg, who, deliveries, archive)
        dest = Path(f"person_{index}.json")
    assert not template.v5hist and template.cp.t == 0
    proof = dict(index=index, who=who, archive=sha(prior), source=sha("source.sha256"),
                 actor=ACTOR, smoke_sha=smoke_sha, labels_opened=False, **payload)
    digest = seal(dest, proof)
    with dest.with_suffix(".sha256").open("x") as f:
        f.write(digest + "\n")
    print(f"PASS F-C14 {mode} {index}: {digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "produce"))
    parser.add_argument("index", type=int, choices=range(9))
    args = parser.parse_args()
    main(args.mode, args.index)
