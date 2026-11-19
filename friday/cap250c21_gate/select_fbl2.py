"""Select current outputs from two complete separately evolved FBL2 branches."""

import json
from pathlib import Path

import numpy as np

from eye_scope import sha
from gate_inputs import held_new, metadata, returning


ORDERS = ("returning", "within0", "full0")
NAMES = ("stock", "fbl2", "selected")


def mask(ramp):
    age = np.arange(len(ramp))
    seen = np.logical_or.accumulate(ramp)
    return age, seen, (age >= 20) & ~seen


def sequence(rows, ramps):
    people, visits, _ = metadata(rows)
    changes = (people[1:] != people[:-1]) | (visits[1:] != visits[:-1])
    blocks = np.split(np.arange(len(rows)), np.flatnonzero(changes) + 1)
    identities = {(str(people[b[0]]), str(visits[b[0]])) for b in blocks}
    if len(identities) != len(blocks):
        raise ValueError("session boundary or delivery interleaving differs")
    age, recognized, eligible = np.empty(len(rows), dtype='i8'), np.empty(len(rows), dtype=bool), np.empty(len(rows), dtype=bool)
    for block in blocks:
        prefix = min(len(block), 31)
        a, r, use = mask(ramps[block])
        check = mask(ramps[block[:prefix]])
        for short, full in zip(check, (a, r, use), strict=True):
            np.testing.assert_array_equal(short, full[:prefix])
        age[block], recognized[block], eligible[block] = a, r, use
    return age, recognized, eligible, len(blocks)


def select(name):
    rows, stock, candidate, ramps, proof = returning() if name == "returning" else held_new(name)
    if len(rows) != len(np.unique(rows)) or any(x.shape != (len(rows), 3) or not np.isfinite(x).all() for x in (stock, candidate)):
        raise ValueError("incomplete stock or FBLight branch")
    age, rec, use, sessions = sequence(rows, ramps)
    known = name == "returning"
    if (len(rows), sessions, int(use.sum())) != ((762, 4, 8) if known else (2655, 14, 2375)):
        raise ValueError("the prespecified causal selector scope differs")
    final = stock.copy()
    final[use] = candidate[use]
    np.testing.assert_array_equal(final[~use], stock[~use])
    np.testing.assert_array_equal(final[use], candidate[use])
    flipping = stock.argmax(1) != candidate.argmax(1)
    if known and (flipping & use).any():
        raise ValueError("the known returning stock decision protection differs")
    path = Path(f"{name}_selected.npz")
    with path.open("xb") as f:
        np.savez_compressed(f, rows=rows, age=age, recognized=rec, eligible=use,
                            stock=stock, fbl2=candidate, selected=final)
    path.chmod(0o444)
    return dict(scope=name, sessions=sessions, rows=len(rows), eligible=int(use.sum()), source=proof,
                selected_changes=int((flipping & use).sum()), suppressed_changes=int((flipping & ~use).sum()),
                recognized=int(rec.sum()), exact_per_session_prefix=True, bank=sha(path))


def main():
    result = {name: select(name) for name in ORDERS}
    path = Path("complete_select.json")
    with path.open("x") as f:
        json.dump(dict(source=sha("source.sha256"), arms=NAMES, scopes=result, held_labels_opened=False,
                       accuracy_computed=False, first_eligible_age=20, emits_after_recognition="stock"),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS labelled-free complete F-C21g selection seal", sha(path), flush=True)


if __name__ == "__main__":
    main()
