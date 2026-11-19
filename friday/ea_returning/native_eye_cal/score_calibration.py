"""Score the authenticated three-arm Eye calibration replay."""

import json
from pathlib import Path

import numpy as np

from eye_cal_scope import TARGET
from eye_scope import DATA, DATA_SHA, PEOPLE, VISITS, pin, sha
from score_eye import DRAWS, SEED, official, point_interval, read_cells
from source_labels import read_training_labels
from verify_calibration import ARMS, deliveries, smokes


NATIVE_SOURCE = "29238af742fed690adc13c025452768e83058fda1f36b6d837b90806bad69891"
CONTRASTS = {"calibration_vs_source": (1, 2), "calibration_vs_production": (0, 2), "source_vs_production": (0, 1)}
WEIGHTS = {"P04": (1., 0.), "P10": (0., 1.), "official_equal": (.5, .5), "20_80": (.2, .8)}


def sealed():
    pin(TARGET / "source.sha256", NATIVE_SOURCE)
    receipt, bank = Path("matched_cal.json"), Path("matched_cal.npz")
    record = json.loads(receipt.read_text())
    seq = deliveries()
    head, prefixes = smokes(seq)
    required = dict(source=NATIVE_SOURCE, order="full0", rows=762, head=head,
                    smokes={str(k): v for k, v in prefixes.items()}, accuracy_computed=False,
                    held_labels_opened=False)
    if (any(record[k] != v for k, v in required.items()) or len(record["parts"]) != 4
            or any(p.stat().st_mode & 0o222 for p in (receipt, bank))):
        raise ValueError("complete calibrated predictions did not seal")
    pin(bank, record["output"])
    for slot, part in enumerate(record["parts"]):
        root = TARGET / f"visit{slot}"
        pin(root / "eye_cal.json", part["receipt"])
        pin(root / "eye_cal.npz", part["predictions"])
        if part["rows"] != len(seq[slot]):
            raise ValueError("per-visit source binding changed")
    with np.load(bank, allow_pickle=False) as z:
        rows = z["rows"]
        np.testing.assert_array_equal(rows, np.concatenate(seq))
        pred = np.column_stack([z[f"{arm}_labels"] for arm in ARMS])
        for i, arm in enumerate(ARMS):
            p = z[f"{arm}_scores"]
            if p.shape != (len(rows), 3) or not np.isfinite(p).all():
                raise ValueError(f"invalid native score for {arm}")
            np.testing.assert_array_equal(pred[:, i], p.argmax(1))
    return rows, pred, seq, sha(receipt)


def summarize(rows, pred, seq, labels, interfaces, metric):
    cells, sessions, cached = read_cells(rows, pred, seq, labels, interfaces, metric)
    for k, record in sessions.items():
        slot = next(i for i, (who, day, _) in enumerate(VISITS) if k == f"{who}|{day}")
        p = cached[slot]["pred"]
        record["changed"] = {name: int((p[:, a] != p[:, b]).sum()) for name, (a, b) in CONTRASTS.items()}
    contrasts = {}
    diagnostics = {}
    for name, (a, b) in CONTRASTS.items():
        rng = np.random.default_rng(SEED)
        contrasts[name] = {key: point_interval(cells[..., [a, b]], w, rng) for key, w in WEIGHTS.items()}
        diagnostics[name] = {}
        for who in PEOPLE:
            role = [cached[j] for j, (person, _, _) in enumerate(VISITS) if person == who]
            ph, y = (np.concatenate([x[k] for x in role]) for k in ("pred", "labels"))
            ok0, ok1 = ph[:, a] == y, ph[:, b] == y
            diagnostics[name][who] = dict(rescued=int((~ok0 & ok1).sum()), lost=int((ok0 & ~ok1).sum()),
                                          changed=int((ph[:, a] != ph[:, b]).sum()))
    return contrasts, diagnostics, sessions


def main():
    rows, pred, seq, blind = sealed()
    metric = official()
    pin(DATA, DATA_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        n, iface = len(z["subject"]), z["interface"][rows]
    q = np.argsort(rows)
    labels = read_training_labels(DATA, rows[q], n)[np.argsort(q)]
    contrasts, diag, sessions = summarize(rows, pred, seq, labels, iface, metric)
    report = dict(scope="FULL-SYSTEM DEV, conditional public returning full0 with fixed original calibration",
                  source=sha("source.sha256"), blind=blind, rows=len(rows), arms=ARMS, official_cells=8,
                  people=PEOPLE, contrasts=contrasts, diagnostics=diag, sessions=sessions,
                  paired_bootstrap=dict(draws=DRAWS, seed=SEED, interval=.8, resampling="people then visits"))
    output = Path("matched_cal_score.json")
    with output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    output.chmod(0o444)
    print("PASS complete calibrated Eye official score", sha(output), contrasts, flush=True)


if __name__ == "__main__":
    main()
