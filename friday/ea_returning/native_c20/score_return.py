"""Score the complete authenticated F-C20r native returning observations."""

import json
from pathlib import Path

import numpy as np

from eye_scope import DATA, DATA_SHA, PEOPLE, VISITS, pin, sha
from pupil_inputs import SOURCE
from score_eye import DRAWS, SEED, official, point_interval, read_cells
from source_labels import read_training_labels
from verify_return import REPLAY, deliveries, smokes


NATIVE_SOURCE = "7b9fc805bc65a9b9f73df20b807f49fb28167fbb25cdcfa31bfad4452f7eaa4e"


def authenticate():
    pin(REPLAY / "source.sha256", NATIVE_SOURCE)
    receipt, bank = Path("matched_c20r.json"), Path("matched_c20r.npz")
    r = json.loads(receipt.read_text())
    sequence = deliveries()
    fixed = dict(source=NATIVE_SOURCE, source_actor=SOURCE, windows=762, order="full0",
                 smokes=smokes(sequence), held_labels_opened=False, accuracy_computed=False)
    if (any(r[k] != v for k, v in fixed.items()) or len(r["parts"]) != 4
            or any(f.stat().st_mode & 0o222 for f in (receipt, bank))):
        raise ValueError("exact F-C20r decision seal did not qualify")
    for slot, part in enumerate(r["parts"]):
        root = REPLAY / "visit" / str(slot)
        pin(root / "return_c20.json", part["receipt"])
        pin(root / "return_c20.npz", part["prediction"])
        if part["rows"] != len(sequence[slot]):
            raise ValueError("the sealed visit scope is incomplete")
    pin(bank, r["bank"])
    with np.load(bank, allow_pickle=False) as z:
        rows = z["rows"]
        np.testing.assert_array_equal(rows, np.concatenate(sequence))
        pred = np.column_stack([z["stock_labels"], z["hard_labels"]])
        for i, arm in enumerate(("stock", "hard")):
            scores = z[f"{arm}_scores"]
            if scores.shape != (762, 3) or not np.isfinite(scores).all():
                raise ValueError("invalid sealed native probabilities")
            np.testing.assert_array_equal(scores.argmax(1), pred[:, i])
    return rows, pred, sequence, sha(receipt)


def summarize(rows, pred, seq, y, groups):
    cells, sessions, cached = read_cells(rows, pred, seq, y, groups, official())
    rng = np.random.default_rng(SEED)
    profiles = {name: point_interval(cells, weight, rng) for name, weight in
                (("P04", (1., 0.)), ("P10", (0., 1.)), ("official_equal", (.5, .5)), ("20_80", (.2, .8)))}
    diagnostics = {}
    for who in PEOPLE:
        selected = [cached[i] for i, (person, _, _) in enumerate(VISITS) if person == who]
        o, gold = (np.concatenate([c[k] for c in selected]) for k in ("pred", "labels"))
        before, after = o[:, 0] == gold, o[:, 1] == gold
        diagnostics[who] = dict(rescued=int((~before & after).sum()), lost=int((before & ~after).sum()),
                                changed=int((o[:, 0] != o[:, 1]).sum()))
    return profiles, sessions, diagnostics


def main():
    rows, pred, seq, proof = authenticate()
    pin(DATA, DATA_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        n, names = len(z["subject"]), z["interface"][rows]
    q = np.argsort(rows)
    y = read_training_labels(DATA, rows[q], n)[np.argsort(q)]
    profile, sessions, diagnostics = summarize(rows, pred, seq, y, names)
    p = dict(scope="FULL-SYSTEM DEV, conditional native returning F-C20r, full0", source=sha("source.sha256"),
             rows=len(rows), official_cells=8, people=PEOPLE, blind=proof, pairings=profile, sessions=sessions,
             diagnostics=diagnostics, actor_excludes_own_visit1=True, original_OOF_exposure_retained=True,
             bootstrap=dict(draws=DRAWS, seed=SEED, interval=.8, method="people and then visits, paired"))
    out = Path("matched_c20r_score.json")
    with out.open("x") as f:
        json.dump(p, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    out.chmod(0o444)
    print("PASS complete F-C20r separate score", sha(out), profile, diagnostics, flush=True)


if __name__ == "__main__":
    main()
