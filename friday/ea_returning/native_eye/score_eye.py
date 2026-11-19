"""Score sealed complete Eye decisions with the official returning-cell metric."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from eye_scope import DATA, DATA_SHA, PEOPLE, REPLAY, REPO, VISITS, pin, sha
from source_labels import read_training_labels
from verify_native import deliveries, verify_smokes


REPLAY_SOURCE = "ae14ac087715cf0383d6f00c63d735ed8fb7e8561959d47602d869e12524b28f"
METRIC = REPO / "2026-competition/benchmark_utils/metrics.py"
METRIC_SHA = "fcc739400806a74259f248618501376a4fe53f2b5a40ccdc20db6a9e2369787d"
DRAWS, SEED = 4000, 460


def sealed():
    pin(REPLAY / "source.sha256", REPLAY_SOURCE)
    receipt, bank = Path("matched_eye.json"), Path("matched_eye.npz")
    p = json.loads(receipt.read_text())
    seq = deliveries()
    smokes = {str(k): v for k, v in verify_smokes(seq).items()}
    if ((p["order"], p["windows"], p["source"], p["native_smokes"]) != ("full0", 762, REPLAY_SOURCE, smokes)
            or p["held_labels_opened"] or p["accuracy_computed"] or len(p["parts"]) != 4
            or any(x.stat().st_mode & 0o222 for x in (receipt, bank))):
        raise ValueError("no complete label-blind Eye decision seal")
    pin(bank, p["output"])
    for slot, info in enumerate(p["parts"]):
        root = REPLAY / f"visit{slot}"
        pin(root / "eye_replay.json", info["receipt"])
        pin(root / "eye_replay.npz", info["output"])
        if info["rows"] != len(seq[slot]):
            raise ValueError("sealed visit scope differs")
    with np.load(bank, allow_pickle=False) as data:
        rows = data["rows"]
        np.testing.assert_array_equal(rows, np.concatenate(seq))
        a = np.column_stack([data[f"{arm}_labels"] for arm in ("stock", "fresh")])
        for j, arm in enumerate(("stock", "fresh")):
            scores = data[f"{arm}_scores"]
            if scores.shape != (762, 3) or not np.isfinite(scores).all():
                raise ValueError("invalid complete native Eye probabilities")
            np.testing.assert_array_equal(a[:, j], scores.argmax(1))
    return rows, a, seq, sha(receipt)


def official():
    pin(METRIC, METRIC_SHA)
    spec = importlib.util.spec_from_file_location("ni26_eye_metrics", METRIC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.balanced_accuracy


def read_cells(rows, pred, seq, labels, interfaces, metric):
    sessions, cells, cached = {}, np.empty((2, 2, 2)), {}
    j = 0
    for slot, ((who, day, count), r) in enumerate(zip(VISITS, seq, strict=True)):
        stop = j + count
        np.testing.assert_array_equal(rows[j:stop], r)
        part, y, group = pred[j:stop], labels[j:stop], interfaces[j:stop]
        local = {}
        for name in sorted(set(group)):
            mask = group == name
            if set(y[mask]) != {0, 1, 2}:
                raise ValueError("unexpected official returning class coverage")
            local[name] = [100 * metric(y[mask], part[mask, k]) for k in range(2)]
        if len(local) != 2:
            raise ValueError("unexpected held interface coverage")
        cells[slot // 2, slot % 2] = np.mean(list(local.values()), axis=0)
        sessions[f"{who}|{day}"] = dict(rows=count, cells=local, changed=int((part[:, 0] != part[:, 1]).sum()))
        cached[slot] = dict(pred=part, labels=y)
        j = stop
    if j != len(rows):
        raise ValueError("incomplete per-visit paired scoring")
    return cells, sessions, cached


def point_interval(cells, people_weights, rng):
    strength = np.asarray(people_weights)
    level = np.einsum("p,pa->a", strength, cells.mean(1))
    ppl = rng.choice(2, size=(DRAWS, 2), p=strength)
    visit = rng.integers(0, 2, size=(DRAWS, 2, 2))
    draw = cells[ppl[..., None], visit].mean((1, 2))
    return dict(stock=float(level[0]), fresh=float(level[1]), pp=float(level[1] - level[0]),
                paired80=np.percentile(draw[:, 1] - draw[:, 0], (10, 90)).tolist())


def main():
    rows, pred, seq, proof_sha = sealed()
    metric = official()
    pin(DATA, DATA_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        people, interfaces = z["subject"], z["interface"][rows]
    io = np.argsort(rows)
    labels = read_training_labels(DATA, rows[io], len(people))[np.argsort(io)]
    cells, sessions, cached = read_cells(rows, pred, seq, labels, interfaces, metric)
    rng = np.random.default_rng(SEED)
    contrasts = {name: point_interval(cells, w, rng) for name, w in (
        (PEOPLE[0], (1., 0.)), (PEOPLE[1], (0., 1.)), ("official_equal", (.5, .5)), ("20_80", (.2, .8)))}
    diag = {}
    for who in PEOPLE:
        role = [cached[j] for j, (name, _, _) in enumerate(VISITS) if who == name]
        p, y = (np.concatenate([entry[k] for entry in role]) for k in ("pred", "labels"))
        a, b = p[:, 0] == y, p[:, 1] == y
        diag[who] = dict(rescued=int((~a & b).sum()), lost=int((a & ~b).sum()),
                         changed=int((p[:, 0] != p[:, 1]).sum()))
    output = Path("matched_eye_score.json")
    result = dict(scope="FULL-SYSTEM DEV, conditional source-fold Eye, returning public P04/P10 full0",
                  source=sha("source.sha256"), blind_receipt=proof_sha, people=PEOPLE, windows=len(rows),
                  old_OOF_calibration_retained=True, official_cells=8, pairings=contrasts, sessions=sessions,
                  diagnostic=diag, bootstrap=dict(draws=DRAWS, seed=SEED, interval=.8,
                  method="sample paired people by specified weights and paired visits within each selected person"))
    with output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    output.chmod(0o444)
    print("PASS separately scored complete eye replay", sha(output), contrasts, flush=True)


if __name__ == "__main__":
    main()
