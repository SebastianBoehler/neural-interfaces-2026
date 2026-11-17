"""Audit existing CMx7-MM metadata/provenance without reading EEG samples."""
import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

TASK_LABEL = {"mentalarithmetic": 0, "motorimagery": 1, "nback": 2}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def block(lo, hi):
    return lo + 1 + 4 * np.arange(max(0, int((hi - lo - 1) // 4)))


def export_log(path):
    """Recover recording-specific rejected channels from the retained exporter log."""
    records, dropped = {}, []
    for line in path.read_text().splitlines():
        if line.startswith("  dropped "):
            assert not dropped, "two channel-drop records without a recording"
            dropped = ast.literal_eval(line.split("dropped ", 1)[1])
        match = re.match(r"(CMx7MM/\d+) (ses-\d+) (\w+) \((\d+), 41, 480\)", line)
        if match:
            person, session, task, n = match.groups()
            key = (person, session, task)
            assert key not in records
            records[key] = {"dropped": dropped, "rows": int(n)}
            dropped = []
        elif line.startswith("skip "):
            dropped = []
    assert not dropped
    return records


def run(root, out, limit):
    t0, cpu0 = time.monotonic(), time.process_time()
    source = root / "data/cmx7mm/windows_1-30.npz"
    rawroot = root / "data/cmx7mm_raw"
    logpath = root / "runs/diag/mental_export_3041064.log"
    log = export_log(logpath)
    out.mkdir(parents=True, exist_ok=False)
    # Deliberately never requests z['X']; NumPy loads archive members lazily.
    with np.load(source) as z:
        d = {key: z[key] for key in
             ("y", "subject", "session", "run", "onset", "ch_names", "interpolated", "classes")}
    keys = sorted(set(zip(d["subject"], d["session"], d["run"])))
    if limit:
        keys = keys[:limit]
    fields = ["row", "person", "session", "recording", "class", "start_s", "end_s",
              "preceding_event", "overlapping_events", "event_start_count", "auditory_overlap_s"]
    rowkeys, counts, eventcounts, seen_rows = set(), Counter(), {}, []
    common_native, recordings = set(d["ch_names"]), []
    with gzip.open(out / "rows.tsv.gz", "wt") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        for person, session, task in keys:
            subject = f"sub-{int(person.split('/')[1]):03d}"
            stem = rawroot / subject / session / "eeg" / f"{subject}_{session}_task-{task}"
            events_path = Path(str(stem) + "_events.tsv")
            events = table(events_path)
            on = np.array([float(e["onset"]) for e in events])
            dur = np.array([float(e["duration"]) for e in events])
            if task == "motorimagery":
                active = on[1:]
            else:
                assert events[0]["trial_type"] == "start"
                active = block(on[0], on[-1] + dur[-1])
            expected = np.r_[block(0, on[0]), active]
            expected_y = np.r_[np.full(len(block(0, on[0])), 3),
                               np.full(len(active), TASK_LABEL[task])]
            ids = np.flatnonzero((d["subject"] == person) & (d["session"] == session)
                                 & (d["run"] == task))
            assert np.array_equal(d["onset"][ids], expected), str(stem)
            assert np.array_equal(d["y"][ids], expected_y), str(stem)
            channels_path = Path(str(stem) + "_channels.tsv")
            channels = [r["name"] for r in table(channels_path)]
            channels = [{"FP1": "Fp1", "FP2": "Fp2"}.get(c, c) for c in channels]
            entry = log[(str(person), str(session), str(task))]
            assert entry["rows"] == len(ids)
            native = set(channels) - set(entry["dropped"])
            common_native &= native
            recording = str(stem.relative_to(rawroot))
            record = {"recording": recording, "rows": len(ids), "dropped": entry["dropped"],
                      "events_sha256": sha(events_path), "channels_sha256": sha(channels_path),
                      "native_target_channels": sorted(native & set(d["ch_names"])),
                      "events": len(events), "event_types": dict(Counter(e["trial_type"] for e in events))}
            recordings.append(record)
            eventcounts.setdefault(str(task), Counter())[len(events)] += 1
            for i in ids:
                start, end = float(d["onset"][i]), float(d["onset"][i] + 4)
                key = (str(person), str(session), str(task), start)
                assert key not in rowkeys
                rowkeys.add(key)
                overlaps = np.flatnonzero((on < end) & (on + dur > start))
                starts = int(((on >= start) & (on < end)).sum())
                overlap_s = float(np.maximum(0, np.minimum(on + dur, end) - np.maximum(on, start)).sum())
                previous = int(np.searchsorted(on, start, side="right") - 1)
                writer.writerow(dict(zip(fields, [int(i), person, session, recording, int(d["y"][i]),
                                                   start, end, previous, ",".join(map(str, overlaps)),
                                                   starts, overlap_s])))
                counts[(str(task), int(d["y"][i]), starts, round(overlap_s, 3))] += 1
                seen_rows.append(int(i))
    if not limit:
        assert sorted(seen_rows) == list(range(len(d["y"])))
    report = {"scope": "metadata/export-log only; no EEG accuracy or EEG samples",
              "audit_script_sha256": sha(Path(__file__)),
              "rows": len(seen_rows), "people": sorted(set(str(d['subject'][i]) for i in seen_rows)),
              "recordings": recordings, "all_row_keys_unique": True,
              "source_sha256": sha(source), "export_log_sha256": sha(logpath),
              "rows_manifest_sha256": sha(out / "rows.tsv.gz"),
              "source_code_sha256": {p: sha(root / p) for p in
                                     ("scripts/export_mental.py", "scripts/export_external.py", "experiments/pretrain.py")},
              "guaranteed_native_target_channels": sorted(common_native),
              "global_interpolation_union": d["interpolated"].tolist(),
              "event_count_distribution": {k: dict(v) for k, v in eventcounts.items()},
              "prompt_overlap_counts": [{"task": k[0], "class": k[1], "event_starts": k[2],
                                         "auditory_overlap_s": k[3], "rows": v} for k, v in sorted(counts.items())],
              "wall_s": time.monotonic() - t0, "cpu_s": time.process_time() - cpu0}
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "recordings"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="real-recording smoke bound")
    args = parser.parse_args()
    run(args.root, args.out, args.limit)
