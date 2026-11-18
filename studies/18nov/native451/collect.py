"""Compare every decision and counter to its paired implementation and archive."""
import hashlib
import json
from pathlib import Path
from statistics import median

BASE = Path("/mnt/lustre/work/gehler/gfh527/ni26_qualify385_v1_20261118")


def read(path):
    return json.loads(path.read_text())


def differences(ref, observed):
    assert ref["rows"] == observed["rows"]
    assert len(ref["rows"]) == len(ref["labels"]) == len(observed["labels"])
    changed = [{"index": i, "row": row, "reference": a, "observed": b}
               for i, (row, a, b) in enumerate(zip(ref["rows"], ref["labels"], observed["labels"]))
               if a != b]
    fields = ref["activity"]
    assert set(fields) == set(observed["activity"]) and len(fields) == 10
    counters = {k: [v, observed["activity"][k]] for k, v in fields.items() if observed["activity"][k] != v}
    return {"labels": changed, "counters": counters}


def main():
    rows = {}
    repeats = {}
    totals = dict(windows=0, native_s=0.0, python_s=0.0, disagreements=0, python_drift=0)
    ratios = []
    for person in range(9):
        archive = read(BASE / f"cap_{person}.json")
        native, python = [read(Path(f"full_{arm}_{person}.json")) for arm in ("native", "python")]
        if person in (4, 5):
            key = next(iter(archive))
            n, p = native.pop("repeat"), python.pop("repeat")
            repeats[key] = dict(native=differences(native[key], n), python=differences(python[key], p),
                                paired=differences(p, n), native_s=n["s"], python_s=p["s"])
        assert list(native) == list(python) == list(archive) and len(archive) == 4
        native_s = python_s = 0.0
        for key, a in archive.items():
            n, p = native[key], python[key]
            paired, drift = differences(p, n), differences(a, p)
            rows[key] = dict(paired=paired, archive_python=drift, n=len(a["rows"]),
                             native_s=n["s"], python_s=p["s"], python_activity=p["activity"])
            totals["windows"] += len(a["rows"])
            totals["disagreements"] += len(paired["labels"])
            totals["python_drift"] += len(drift["labels"])
            native_s += n["s"]
            python_s += p["s"]
        ratios.append(native_s / python_s)
        totals["native_s"] += native_s
        totals["python_s"] += python_s
    assert totals["windows"] == 4260 and len(rows) == 36
    result = dict(totals=totals, median_paired_time_ratio=median(ratios), person_time_ratios=ratios,
                  streams=rows, warm_first_stream=repeats,
                  input_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in sorted(Path(".").glob("full_*.json"))})
    Path("result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(totals, median(ratios), flush=True)


if __name__ == "__main__":
    main()
