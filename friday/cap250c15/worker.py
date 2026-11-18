"""Run matched full cap250 sessions with the outer-tail rebalancer intervention."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from bank_tools import archive_parity, seal, sha
from benchmark_utils.streaming import fresh_model
from rebalance import EXPOSURE_TOL, Outer
from settings import ARMS, SEEDS, STOCK_WORK, schedule


TOL = 2e-5
PUBLIC = Path("/native/reference_normal.json")


def gate(g):
    return (-1., 0., 0.) if g is None else tuple(float(v) for v in g)


def run(template, eeg, rows, arm):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    observer = Outer(model, arm)
    labels, posterior, v5, gates, targets, warm, means = [], [], [], [], [], [], []
    keys = sorted(model.ft.member)
    for row in rows:
        row = int(row)
        out = model.predict(eeg[row:row+1].to("cuda:0"))
        assert out.device.type == "cpu" and out.dtype == torch.int64 and out.shape == (1,)
        assert not model.dead and not model.cp.dead and not model.cf.dead
        assert 0 <= int(out[0]) <= 2
        trace = model.trace
        labels.append(int(out[0]))
        posterior.append(np.r_[observer.x, observer.em_in, trace["out"]])
        v5.append(np.r_[trace["mix"], trace["shadow"], trace["out_v5"]])
        gates.append((gate(trace["contrast"]), gate(trace["identity"])))
        targets.append(trace["v6"]["label"].copy())
        fine = trace["v6"]["p_ft"]
        warm.append(np.asarray([fine.get(k, [0., 0., 0.]) for k in keys]))
        means.append(int(model.recB))
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    activity = replay.activity(model)
    assert activity["windows"] == len(labels) and activity["errors"] == activity["ft_over"] == 0
    assert activity["ft_steps"] == activity["ft_refits"] * 20
    assert model.routeB.log["n"] == len(labels) and model.routeB.log["err"] == 0
    got = observer.receipt()
    assert got["calls"] == got["em_calls"] == len(labels)
    assert got["rb_calls"] == (0 if arm == "stock" else len(labels))
    assert got["exposure_max_error"] <= EXPOSURE_TOL
    work = [activity[k] for k in STOCK_WORK]
    info = dict(labels=labels, activity=activity, seconds=seconds, receipt=got,
                own_staged=model.routeB.log["staged"], own_routed=model.routeB.log["routed"], protected_work=work)
    arrays = dict(v5=v5, gate=gates, targets=targets, warm=warm, latch=means, posterior=posterior)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    arrays["ft_lab"] = np.asarray(model.ft.lab)
    arrays["ft_weights"] = np.asarray(model.ft.wt)
    assert all(np.isfinite(v).all() for v in arrays.values())
    return info, arrays


def diff(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    return float(np.abs(x-y).max()) if x.size else 0.


def protected(reference, candidate):
    a, x = reference
    b, y = candidate
    assert a["protected_work"] == b["protected_work"]
    assert a["own_staged"] == b["own_staged"] and a["own_routed"] == b["own_routed"]
    assert np.array_equal(x["latch"], y["latch"]) and np.array_equal(x["ft_lab"], y["ft_lab"])
    assert np.array_equal(x["gate"][:, :, 0], y["gate"][:, :, 0])
    assert np.array_equal(x["gate"][:, :, 2] > 0, y["gate"][:, :, 2] > 0)
    for k in ("v5", "warm", "targets"):
        assert np.array_equal(x[k].reshape(len(a["labels"]), -1, 3).argmax(-1),
                              y[k].reshape(len(a["labels"]), -1, 3).argmax(-1)), k
    names = ("v5", "gate", "targets", "warm", "ft_weights")
    errors = {k: diff(x[k], y[k]) for k in names}
    errors["outer_mix"] = diff(x["posterior"][:, :3], y["posterior"][:, :3])
    assert max(errors.values()) <= TOL, errors
    return errors


def repetitions(arm, group, other, model, eeg, results):
    run(model, eeg, other[:31], arm)
    prefix, p = run(model, eeg, group[:31], arm)
    full, q = results[arm]
    n = len(prefix["labels"])
    assert prefix["labels"] == full["labels"][:n]
    error = diff(p["posterior"], q["posterior"][:n])
    assert error <= TOL and np.array_equal(p["latch"], q["latch"][:n])
    assert np.array_equal(p["posterior"][:, :6].reshape(n, 2, 3).argmax(-1),
                          q["posterior"][:n, :6].reshape(n, 2, 3).argmax(-1))
    return error


def grade_smoke(model, eeg, rows, other, archive):
    pre, _ = run(model, eeg, rows, "stock")
    archive_parity(pre, archive)
    results = {a: run(model, eeg, rows, a) for a in ("stock", "identity", "scalar", "robust")}
    stock = results["stock"]
    stock_m = stock[1]["posterior"]
    for a in ("stock", "identity"):
        archive_parity(results[a][0], archive)
    control = diff(stock_m, results["identity"][1]["posterior"])
    assert control <= TOL
    print(f"PASS archive and identity control, max posterior error {control:.7g}", flush=True)
    report = {}
    for a, (info, tape) in results.items():
        assert info["activity"]["recognized"] == -1 and info["activity"]["ft_refits"] > 0
        error = protected(stock, (info, tape))
        assert diff(stock_m[0], tape["posterior"][0]) <= TOL
        interleaved = repetitions(a, rows, other, model, eeg, results)
        report[a] = dict(**info, protected_max=error, reset_prefix_max=interleaved)
    r, s = (results[a] for a in ("robust", "scalar"))
    assert r[0]["receipt"]["above_001"] >= 12
    assert r[0]["receipt"]["clipped_history_occurrences"] > 0
    assert diff(r[1]["posterior"][:, 3:6], s[1]["posterior"][:, 3:6]) > .001
    assert diff(r[1]["posterior"][:, 3:6], stock_m[:, 3:6]) > .001
    return report


def recognized():
    eeg, all_streams, ch = replay.read_source()
    model = replay.load(Path("/package"), ch)
    rows = dict(all_streams)["P10-2"][:31]
    expected = json.loads(PUBLIC.read_text())["streams"]["first"]["P10-2"]
    results = {a: run(model, eeg, rows, a) for a in ("stock", "identity", "scalar", "robust")}
    stock = results["stock"]
    assert stock[0]["labels"] == expected["labels"][:31]
    assert results["identity"][0]["labels"] == stock[0]["labels"]
    report = {}
    for a, (info, tape) in results.items():
        assert info["activity"]["recognized"] == expected["activity"]["recognized"] == 10
        assert info["activity"]["ft_refits"] == 0 and info["own_staged"] == info["own_routed"] > 0
        report[a] = dict(**info, protected_max=protected(stock, (info, tape)))
    return dict(rows=rows.tolist(), result=report, archive=sha(PUBLIC))


def produce(index, model, eeg, who, deliveries, archive):
    visits = [f"{who}|{v}" for v in (1, 2)]
    warm = deliveries[411][visits[0]]
    controls = {}
    for a in ARMS:
        got, _ = run(model, eeg, warm, a)
        if a == "stock":
            key = visits[0]+"|411"
            archive_parity(got, archive[key])
            controls[key] = dict(rows=warm.tolist(), run=got)
    pre = deliveries[1411][visits[0]]
    got, _ = run(model, eeg, pre, "stock")
    key = visits[0]+"|1411"
    archive_parity(got, archive[key])
    controls[key] = dict(rows=pre.tolist(), run=got)
    out = {}
    for s, seed in enumerate(SEEDS):
        for d, visit in enumerate(visits):
            rows = deliveries[seed][visit]
            rotate = (index + 2*s + d) % len(ARMS)
            seq = ARMS[rotate:] + ARMS[:rotate]
            all_arms = {a: run(model, eeg, rows, a) for a in seq}
            summary = {a: dict(**all_arms[a][0], protected_max=protected(all_arms["stock"], all_arms[a])) for a in seq}
            key = f"{visit}|{seed}"
            out[key] = dict(rows=rows.tolist(), order=seq, arms=summary)
            print(f"SEALED {key}: " + " ".join(f"{a} {summary[a]['seconds']:.2f}s" for a in seq), flush=True)
    assert sum(s["arms"]["robust"]["receipt"]["above_001"] for s in out.values()) > 0
    return dict(sessions=out, archived_controls=controls)


def main(mode, index):
    assert torch.__version__ == "2.14.1+cu130" and "A100" in torch.cuda.get_device_name()
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    with np.load("/external.npz", allow_pickle=False) as f:
        eeg = torch.from_numpy(f["X"])
        people, visits, ch = (f[k] for k in ("subject", "session", "ch_names"))
    assert eeg.shape == (2130, 41, 480) and eeg.dtype == torch.float32 and torch.isfinite(eeg).all()
    who = sorted(set(people))[index]
    deliveries = schedule(people, visits)
    parent = Path(f"/prior/cap_{index}.json")
    archive = json.loads(parent.read_text())
    for name, original in archive.items():
        stream, seed = name.rsplit("|", 1)
        assert deliveries[int(seed)][stream].tolist() == original["rows"]
    model = replay.load(Path("/package"), ch.tolist())
    assert model.perm is None and model.routed == 1 and model.ft.on and model.ft.cap == 250
    source = sha("source.sha256")
    if mode == "smoke":
        assert index == 5 and who == "Scherer2015/G"
        key = who+"|1|411"
        rows, other = (deliveries[411][f"{who}|{day}"] for day in (1, 2))
        assert len(rows) == 120 and rows[0] == 1257 and other[0] == 1383
        evidence = grade_smoke(model, eeg, rows, other, archive[key])
        evidence = dict(arms=evidence, rows=rows.tolist(), recognized_public=recognized(), passed=True)
        smoke_sha = None
        dest = Path("smoke.json")
    else:
        smoke = Path("smoke.json")
        smoke_sha = sha(smoke)
        r = json.loads(smoke.read_text())
        assert r["passed"] and r["source"] == source and not r["labels_opened"]
        assert smoke.with_suffix(".sha256").read_text().strip() == smoke_sha
        evidence = produce(index, model, eeg, who, deliveries, archive)
        dest = Path(f"person_{index}.json")
    assert not model.v5hist and model.cp.t == 0
    evidence.update(index=index, who=who, smoke_sha=smoke_sha, labels_opened=False, source=source, archive=sha(parent))
    digest = seal(dest, evidence)
    with dest.with_suffix(".sha256").open("x") as f:
        f.write(digest + "\n")
    print(f"PASS F-C15 {mode} {index}: {digest}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("smoke", "produce"))
    p.add_argument("index", type=int, choices=range(9))
    options = p.parse_args()
    main(options.mode, options.index)
