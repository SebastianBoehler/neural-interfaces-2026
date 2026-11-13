"""Compare two recordings of lean/tools/record.py window by window.

Usage: python lean/tools/parity.py REF.npz NEW.npz [--out JSON]

For every case: the number of windows, how many were routed by each gate and staged, the people routed to and how
often the routed person changed, whether the labels are equal, and per recorded field whether the bytes are equal
and the largest absolute difference. Exit status 0 only if every field of every window is bitwise equal.
"""
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = ("label", "out", "mix", "core1", "shadow", "u", "contrast", "identity", "stage")


def describe(r):
    """What happened in one case (from one recording)."""
    ramp = lambda a: np.nan_to_num(a[:, 2])
    person = np.where(np.isnan(r["stage"][:, 0]), -1, r["stage"][:, 0]).astype(int)
    routed = person[person >= 0]
    return dict(windows=len(person), identity_gate=int((ramp(r["identity"]) > 0).sum()),
                contrast_gate=int((ramp(r["contrast"]) > 0).sum()), staged=len(routed),
                people=sorted(set(routed.tolist())), person_changes=int((np.diff(routed) != 0).sum()))


def main():
    ref, new = (dict(np.load(p)) for p in sys.argv[1:3])
    assert np.array_equal(ref["case"], new["case"]), "the recordings cover different cases or windows"
    report, exact = {}, True
    for case in dict.fromkeys(ref["case"]):
        rows = ref["case"] == case
        a, b = ({k: r[k][rows] for k in FIELDS} for r in (ref, new))
        info = describe(a)
        info["labels_equal"] = bool(np.array_equal(a["label"], b["label"]))
        info["bitwise"] = {k: a[k].tobytes() == b[k].tobytes() for k in FIELDS}
        diff = {k: np.abs(np.nan_to_num(a[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                          - np.nan_to_num(b[k], nan=0.0, posinf=0.0, neginf=0.0)) for k in FIELDS}
        info["max_abs_diff"] = {k: float(v.max()) for k, v in diff.items()}
        info["same_gaps"] = all(np.array_equal(np.isnan(a[k]), np.isnan(b[k])) for k in FIELDS if a[k].dtype.kind == "f")
        info["seconds"] = [round(float(r["seconds"][rows].sum()), 1) for r in (ref, new)]
        ok = all(info["bitwise"].values())
        exact &= ok
        report[str(case)] = info
        print(f"{case:22s} windows {info['windows']:4d}  identity {info['identity_gate']:4d}  contrast "
              f"{info['contrast_gate']:4d}  staged {info['staged']:4d}  people {info['people']}  changes "
              f"{info['person_changes']}  labels {'equal' if info['labels_equal'] else 'DIFFER'}  "
              f"{'bitwise equal' if ok else 'max diff %.3g' % max(info['max_abs_diff'].values())}")
    total = dict(windows=int(len(ref["case"])), cases=len(report), bitwise_equal=bool(exact),
                 labels_equal=bool(np.array_equal(ref["label"], new["label"])),
                 max_abs_diff_posterior=max(r["max_abs_diff"]["out"] for r in report.values()))
    print(json.dumps(total))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(dict(total=total, cases=report), indent=1) + "\n")
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
