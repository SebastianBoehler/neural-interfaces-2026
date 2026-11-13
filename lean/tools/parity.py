"""Compare two recordings of lean/tools/record.py window by window.

Usage: python lean/tools/parity.py REF.npz NEW.npz [--out JSON]

Cases present in only one recording are listed and skipped. For every common case: the number of windows, how
many were routed by each gate and staged, the people routed to and how often the routed person changed, whether
the labels are equal, and per recorded field whether the bytes are equal and the largest absolute difference.
Exit status 0 only if every field of every window is bitwise equal.
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
    common = [c for c in dict.fromkeys(ref["case"]) if c in set(new["case"])]
    skipped = sorted(set(ref["case"]) ^ set(new["case"]))
    assert common, "the recordings share no case"
    report, exact = {}, True
    for case in common:
        a, b = ({k: r[k][r["case"] == case] for k in (*FIELDS, "seconds")} for r in (ref, new))
        assert len(a["label"]) == len(b["label"]), f"{case}: the recordings hold different numbers of windows"
        info = describe(a)
        info["labels_equal"] = bool(np.array_equal(a["label"], b["label"]))
        info["bitwise"] = {k: a[k].tobytes() == b[k].tobytes() for k in FIELDS}
        diff = {k: np.abs(np.nan_to_num(a[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                          - np.nan_to_num(b[k], nan=0.0, posinf=0.0, neginf=0.0)) for k in FIELDS}
        info["max_abs_diff"] = {k: float(v.max()) for k, v in diff.items()}
        info["same_gaps"] = all(np.array_equal(np.isnan(a[k]), np.isnan(b[k]))
                                for k in FIELDS if a[k].dtype.kind == "f")
        info["seconds"] = [round(float(r["seconds"].sum()), 1) for r in (a, b)]
        ok = all(info["bitwise"].values())
        exact &= ok
        report[str(case)] = info
        print(f"{case:22s} windows {info['windows']:4d}  identity {info['identity_gate']:4d}  contrast "
              f"{info['contrast_gate']:4d}  staged {info['staged']:4d}  people {info['people']}  changes "
              f"{info['person_changes']}  labels {'equal' if info['labels_equal'] else 'DIFFER'}  "
              f"{'bitwise equal' if ok else 'max diff %.3g' % max(info['max_abs_diff'].values())}")
    total = dict(windows=sum(r["windows"] for r in report.values()), cases=len(report), cases_in_one_only=skipped,
                 bitwise_equal=bool(exact), labels_equal=all(r["labels_equal"] for r in report.values()),
                 max_abs_diff_posterior=max(r["max_abs_diff"]["out"] for r in report.values()))
    print(json.dumps(total))
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.write_text(json.dumps(dict(total=total, cases=report), indent=1) + "\n")
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
