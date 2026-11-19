"""Read only the four official F-C21r held visits after their complete seal."""

import argparse
import json
from pathlib import Path

import numpy as np

from eye_scope import DATA, DATA_SHA, PEOPLE, VISITS, pin, sha
from score_eye import DRAWS, SEED, official, point_interval, read_cells
from source_labels import read_training_labels
from verify_fbl2 import ARMS, BANK, HOOK, deliveries, output, smokes


def authenticated(source):
    rows = deliveries()
    receipt = BANK.with_suffix(".json")
    proof = json.loads(receipt.read_text())
    fixed = dict(source=source, hook=HOOK, order="full0", windows=762, smokes=smokes(rows, source),
                 held_labels_opened=False, accuracy_computed=False)
    if (any(proof[k] != v for k, v in fixed.items()) or len(proof["parts"]) != 4
            or any(x.stat().st_mode & 0o222 for x in (receipt, BANK))):
        raise ValueError("the complete returning FBLight seal did not qualify")
    for i, part in enumerate(proof["parts"]):
        p = output(i, False)
        pin(p, part["bank"])
        pin(p.with_suffix(".json"), part["receipt"])
        if part["rows"] != len(rows[i]):
            raise ValueError("the signed visit is incomplete")
    pin(BANK, proof["bank"])
    with np.load(BANK, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["rows"], np.concatenate(rows))
        pred = np.column_stack([z[f"{arm}_labels"] for arm in ARMS])
        for k, arm in enumerate(ARMS):
            p = z[f"{arm}_scores"]
            if p.shape != (762, 3) or not np.isfinite(p).all():
                raise ValueError("incomplete sealed F-C21r decisions")
            np.testing.assert_array_equal(p.argmax(1), pred[:, k])
        return z["rows"], pred, rows, sha(receipt)


def score(source):
    rows, pred, order, receipt = authenticated(source)
    metric = official()
    pin(DATA, DATA_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        n, interfaces = len(z["subject"]), z["interface"][rows]
    io = np.argsort(rows)
    labels = read_training_labels(DATA, rows[io], n)[np.argsort(io)]
    cells, sessions, cache = read_cells(rows, pred, order, labels, interfaces, metric)
    rng = np.random.default_rng(SEED)
    pairings = {name: point_interval(cells, weights, rng) for name, weights in
                (("P04", (1., 0.)), ("P10", (0., 1.)), ("official_equal", (.5, .5)), ("20_80", (.2, .8)))}
    diag = {}
    for who in PEOPLE:
        parts = [cache[i] for i, (person, _, _) in enumerate(VISITS) if who == person]
        p, y = (np.concatenate([x[k] for x in parts]) for k in ("pred", "labels"))
        s, h = p[:, 0] == y, p[:, 1] == y
        diag[who] = dict(rescued=int((~s & h).sum()), lost=int((s & ~h).sum()), changed=int((p[:, 0] != p[:, 1]).sum()))
    result = dict(scope="FULL-SYSTEM DEV conditional returning F-C21r", windows=len(rows), order="full0",
                  official_cells=8, native=source, source=sha("source.sha256"), blind=receipt, people=PEOPLE,
                  retained_original_gallery_and_production_auxiliaries=True, pairings=pairings, sessions=sessions,
                  diagnostic=diag, bootstrap=dict(draws=DRAWS, seed=SEED, interval=.8, method="people then visits, paired"))
    path = Path("matched_c21r_score.json")
    with path.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS separate complete F-C21r score", sha(path), pairings, diag, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("native_source")
    score(cli.parse_args().native_source)
