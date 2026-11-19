"""Exercise the actual input stream first, then run the full cap250 worker."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import replay
from bank_tools import archive_parity, seal, sha
from benchmark_utils.streaming import fresh_model
from c12_history import ALIGNED
from fractional import install
from settings import ACTIVE, ARMS, SEEDS, schedule


def match_exposure(first, second, n):
    assert first["mode"] == "informative" and second["mode"] == "permuted"
    assert first["rows"] == second["rows"] == n and first["prior_sha"] == second["prior_sha"]
    np.testing.assert_array_equal(first["weights"], second["weights"])
    error = float(np.abs(np.asarray(first["branch_mass"])-second["branch_mass"]).max())
    assert error <= 2e-12
    assert first["moved"] == 0 and second["moved"] == 4*(n//4)
    assert first["blocks"] == second["blocks"] == n//4
    return error


def stream_only(template, eeg, rows, arm, other):
    model = fresh_model(template)
    stream = install(model, arm)

    def aligned(index):
        return np.stack([stream(eeg[int(i)].numpy()) for i in index])

    outputs = aligned(rows)
    stats = None if arm == "stock" else stream.receipt()
    stream.reset_state()
    aligned(other[:31])
    stream.reset_state()
    np.testing.assert_array_equal(aligned(rows[:31]), outputs[:31])
    return outputs, stats


def preflight(template, eeg, rows, other):
    banks = {a: stream_only(template, eeg, rows, a, other) for a in ("stock", "disabled", *ACTIVE)}
    stock, disabled = (banks[a][0] for a in ("stock", "disabled"))
    first, second = (banks[a] for a in ("informative", "permuted"))
    np.testing.assert_array_equal(stock, disabled)
    np.testing.assert_array_equal(first[0][0], stock[0])
    np.testing.assert_array_equal(second[0][0], stock[0])
    np.testing.assert_array_equal(first[0][:4], second[0][:4])
    error = match_exposure(first[1], second[1], len(rows))
    span = float(np.ptp(first[1]["weights"]))
    magnitude = np.abs(first[0]-second[0]).max(axis=(1, 2))
    changed = int(np.count_nonzero(magnitude > 1e-3))
    assert span > .10 and first[1]["separation"] > .02 and changed >= 30, (span, first[1]["separation"], changed)
    return dict(first_active_rows=changed, weight_span=span, branch_separation=first[1]["separation"],
                max_paired_input_delta=float(magnitude.max()), exposure_max_error=error,
                disabled_bitwise=True, first_window_exact=True, pending_first_block_exact=True, prefix_reset_exact=True)


def worker(template, eeg, rows, arm, *, trace=False):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    stream = install(model, arm)
    labels, aligned, terminal = [], [], []
    last = None
    cp = model.cp.step

    def scores(*args):
        nonlocal last
        q = cp(*args)
        last = q.copy()
        return q

    model.cp.step = scores
    digest = hashlib.sha256()
    for row in rows:
        i = int(row)
        p = model.predict(eeg[i:i+1].to("cuda:0"))
        assert p.shape == (1,) and p.dtype == torch.int64 and p.device.type == "cpu"
        assert not model.dead and not model.cp.dead and not model.cf.dead
        assert last is not None and np.isfinite(last).all() and int(last.argmax()) == int(p[0])
        x = model.nets.last_inputs[ALIGNED[0]]
        assert x.shape == (41,480) and x.dtype == np.float32 and np.isfinite(x).all()
        assert all(model.nets.last_inputs[j] is x for j in ALIGNED)
        digest.update(np.ascontiguousarray(x))
        if trace:
            aligned.append(x.copy())
            terminal.append(last.copy())
        last = None
        labels.append(int(p[0]))
    torch.cuda.synchronize()
    seconds = perf_counter()-start
    activity = replay.activity(model)
    assert activity["windows"] == len(rows) and activity["errors"] == activity["ft_over"] == 0
    assert activity["ft_steps"] == 20*activity["ft_refits"]
    stat = None if arm == "stock" else stream.receipt()
    assert stat is None or stat["rows"] == len(rows)
    info = dict(labels=labels, activity=activity, seconds=seconds, aligned_sha=digest.hexdigest(), history=stat)
    tape = (np.asarray(aligned), np.asarray(terminal)) if trace else None
    return info, tape


def smoke(template, eeg, rows, other, archive):
    proof = preflight(template, eeg, rows, other)
    print("PASS F-C17 standalone actual alignment", proof, flush=True)
    results = {a: worker(template, eeg, rows, a, trace=True) for a in ("stock", "disabled", *ACTIVE)}
    a, b = (results[v] for v in ("stock", "disabled"))
    archive_parity(a[0], archive)
    archive_parity(b[0], archive)
    np.testing.assert_array_equal(a[1][0], b[1][0])
    i, p = (results[v] for v in ("informative","permuted"))
    assert match_exposure(i[0]["history"], p[0]["history"], len(rows)) <= 2e-12
    np.testing.assert_array_equal(i[1][0][:4], p[1][0][:4])
    assert i[0]["labels"][0] == p[0]["labels"][0] == a[0]["labels"][0]
    max_delta = float(np.abs(i[1][1]-p[1][1]).max())
    assert max_delta > 1e-3
    summary = {}
    for arm, (full, tape) in results.items():
        assert full["activity"]["ft_refits"] > 0
        np.testing.assert_array_equal(tape[0][0], a[1][0][0])
        worker(template, eeg, other[:31], arm)
        short, prefix = worker(template, eeg, rows[:31], arm, trace=True)
        assert short["labels"] == full["labels"][:31]
        np.testing.assert_array_equal(prefix[0], tape[0][:31])
        if arm in ACTIVE:
            np.testing.assert_array_equal(short["history"]["weights"], full["history"]["weights"][:31])
            np.testing.assert_array_equal(short["history"]["branch_mass"], full["history"]["branch_mass"][:31])
        summary[arm] = dict(**full, max_prefix_probability_error=float(np.abs(prefix[1]-tape[1][:31]).max()),
                            changes=int(np.count_nonzero(np.asarray(full["labels"]) != a[0]["labels"])))
    return dict(passed=True, rows=rows.tolist(), preflight=proof, full_terminal_delta=max_delta, arms=summary)


def produce(index, template, eeg, person, deliveries, archive):
    visits = [f"{person}|{v}" for v in (1, 2)]
    start = deliveries[411][visits[0]]
    anchors = {}
    for a in ARMS:
        info, _ = worker(template, eeg, start, a)
        if a == "stock":
            key = visits[0]+"|411"
            archive_parity(info, archive[key])
            anchors[key] = dict(rows=start.tolist(), run=info)
    rows = deliveries[1411][visits[0]]
    key = visits[0]+"|1411"
    info, _ = worker(template, eeg, rows, "stock")
    archive_parity(info, archive[key])
    anchors[key] = dict(rows=rows.tolist(), run=info)
    output = {}
    for q, seed in enumerate(SEEDS):
        for d, visit in enumerate(visits):
            rows = deliveries[seed][visit]
            rot = (index + 2*q + d) % len(ARMS)
            seq = ARMS[rot:]+ARMS[:rot]
            results = {a: worker(template, eeg, rows, a)[0] for a in seq}
            fit, ctl = (results[a]["history"] for a in ("informative","permuted"))
            error = match_exposure(fit, ctl, len(rows))
            assert results["informative"]["aligned_sha"] != results["permuted"]["aligned_sha"]
            entry = dict(rows=rows.tolist(), order=seq, arms=results, exposure_error=error)
            output[f"{visit}|{seed}"] = entry
            print(f"SEALED F-C17 {visit}|{seed}: "+", ".join(f"{a} {results[a]['seconds']:.2f}s" for a in seq), flush=True)
    return dict(sessions=output, archived_controls=anchors)


def main(mode, index):
    assert torch.__version__ == "2.14.1+cu130" and "A100" in torch.cuda.get_device_name()
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    with np.load("/external.npz", allow_pickle=False) as f:
        eeg = torch.from_numpy(f["X"])
        people, days, channels = (f[k] for k in ("subject","session","ch_names"))
    assert eeg.shape == (2130,41,480) and eeg.dtype == torch.float32 and torch.isfinite(eeg).all()
    who = sorted(set(people))[index]
    orders = schedule(people, days)
    old = Path(f"/prior/cap_{index}.json")
    archived = json.loads(old.read_text())
    for key, data in archived.items():
        visit, seed = key.rsplit("|",1)
        assert orders[int(seed)][visit].tolist() == data["rows"]
    model = replay.load(Path("/package"), channels.tolist())
    assert model.perm is None and model.routed == 1 and model.ft.on and model.ft.cap == 250
    own_src = sha("source.sha256")
    if mode == "smoke":
        assert index == 5 and who == "Scherer2015/G"
        rows = orders[411][who+"|1"]
        other = orders[411][who+"|2"]
        assert len(rows) == 120 and (rows[0],other[0]) == (1257,1383)
        output = smoke(model,eeg,rows,other,archived[who+"|1|411"])
        smoke_sha = None
        dest = Path("smoke.json")
    else:
        smoke_file = Path("smoke.json")
        smoke_sha = sha(smoke_file)
        proof = json.loads(smoke_file.read_text())
        assert proof["passed"] and proof["source"] == own_src and not proof["labels_opened"]
        assert Path("smoke.sha256").read_text().strip() == smoke_sha
        output = produce(index,model,eeg,who,orders,archived)
        dest = Path(f"person_{index}.json")
    assert not model.v5hist and model.cp.t == 0
    output.update(index=index, who=who, source=own_src, archive=sha(old), smoke_sha=smoke_sha, labels_opened=False)
    signature = seal(dest,output)
    with dest.with_suffix(".sha256").open("x") as f:
        f.write(signature+"\n")
    print(f"PASS F-C17 {mode} {who}: {signature}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("smoke","produce"))
    p.add_argument("index", type=int, choices=range(9))
    opts = p.parse_args()
    main(opts.mode, opts.index)
