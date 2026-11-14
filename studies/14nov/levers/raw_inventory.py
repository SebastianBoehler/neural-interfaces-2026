"""s248 M2 / M3(i) / M5: inventory of the raw PROTEUS release on disk (no download).
Per run: EDF start date/time, duration, digital resolution of channel 1, SHA-256 of the signal payload (duplicates);
per cued run: trial task duration (cue to end_trial), inter-trial gap, 4-s crops at 0.5-s steps inside the task period;
per rest run: seconds and non-overlapping 4-s windows.
Usage: python studies/14nov/levers/raw_inventory.py"""
import csv, glob, hashlib, os, re
from collections import defaultdict
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
ROOT = REPO + "/data/neural_compet/Dreyer2026Proteus/download/nm000290"


def edf_header(path):
    with open(path, "rb") as f:
        h = f.read(256); ns = int(h[252:256]); hb = int(h[184:192]); nrec = int(h[236:244]); dur = float(h[244:252])
        sh = f.read(ns * 256); f.seek(hb); payload = hashlib.sha256(f.read()).hexdigest()
    pmin = float(sh[ns * 104: ns * 104 + 8])  # signal header fields are blocks of ns entries; channel 1 of each
    pmax, dmin, dmax = float(sh[ns * 112: ns * 112 + 8]), float(sh[ns * 120: ns * 120 + 8]), float(sh[ns * 128: ns * 128 + 8])
    return dict(date=h[168:176].decode(), time=h[176:184].decode(), sec=nrec * dur, res=(pmax - pmin) / (dmax - dmin), sha=payload)


rows, hashes = [], defaultdict(list)
for edf in sorted(glob.glob(ROOT + "/sub-*/ses-*/eeg/*_eeg.edf")):
    m = re.search(r"sub-(P\d+)_ses-(\d)_task-(\w+)_run-(\d)", edf); sub, ses, task, run = m.groups(); hd = edf_header(edf)
    hashes[hd["sha"]].append(os.path.basename(edf))
    ev = list(csv.DictReader(open(edf.replace("_eeg.edf", "_events.tsv")), delimiter="\t"))
    cues = [float(e["onset"]) for e in ev if e["trial_type"] in ("mi", "sub", "word")]
    ends = [float(e["onset"]) for e in ev if e["trial_type"] == "end_trial"]
    starts = [float(e["onset"]) for e in ev if e["trial_type"] == "start_trial"]
    task_len = np.median([min([t for t in ends if t > c] or [np.nan]) - c for c in cues]) if cues else np.nan
    gap = np.median(np.diff(cues)) if len(cues) > 1 else np.nan
    rows.append(dict(sub=sub, ses=ses, task=task, run=run, ncue=len(cues), task_len=task_len, cue_gap=gap, **hd))

print("s248 raw inventory")
print("\n1. Duplicate signal payloads (SHA-256 over all data records):")
dup = [v for v in hashes.values() if len(v) > 1]
print(f"   runs {len(rows)}, distinct payloads {len(hashes)}, duplicated groups {len(dup)}", *dup, sep="\n   ")
print("\n2. Rest runs (eyes open / closed): seconds and non-overlapping 4-s windows")
rest = [r for r in rows if r["task"].startswith("Baseline")]
for r in rest:
    print(f"   {r['sub']} s{r['ses']} {r['task']:10s} {r['sec']:7.1f} s  -> {int(r['sec'] // 4):4d} windows")
print(f"   total rest {sum(r['sec'] for r in rest) / 60:.1f} min -> {sum(int(r['sec'] // 4) for r in rest)} windows "
      f"(per session {sum(int(r['sec'] // 4) for r in rest) / 14:.0f})")
print("\n3. Cued runs: task duration (cue -> end_trial), cue-to-cue gap, crops")
cued = [r for r in rows if r["ncue"]]
tl = np.array([r["task_len"] for r in cued]); gp = np.array([r["cue_gap"] for r in cued]); nc = sum(r["ncue"] for r in cued)
print(f"   cued runs {len(cued)}, cues {nc}; task duration median {np.nanmedian(tl):.2f} s (range {np.nanmin(tl):.2f}-{np.nanmax(tl):.2f});"
      f" cue gap median {np.nanmedian(gp):.2f} s")
L = np.nanmedian(tl); ncrop = int(round((L - 4) / 0.5)) + 1
print(f"   4-s crops fully inside the task period at 0.5-s steps: {ncrop} per trial (the official crop is cue+1 s) -> "
      f"{ncrop * nc} rows ({ncrop}x); excluding the cue+0..1 s visual-cue second: {int(round((L - 5) / 0.5)) + 1} per trial")
pre = np.nanmedian(gp) - L
print(f"   between end_trial and the next cue: {pre:.1f} s (fixation, beep; REST4 already uses it)")
print(f"   cued-run recording total {sum(r['sec'] for r in cued) / 60:.0f} min; windows used officially {nc} x 4 s = {nc * 4 / 60:.0f} min "
      f"({100 * nc * 4 / sum(r['sec'] for r in cued):.0f} % of the cued recording)")
print("\n4. EDF resolution (uV per bit, channel 1) per run: distinct values")
res = defaultdict(list)
for r in rows:
    res[round(r["res"], 4)].append(f"{r['sub']}s{r['ses']}{r['task'][:9]}{r['run']}")
for k, v in sorted(res.items()):
    print(f"   {k:8.4f}: {len(v)} runs" + (f"  {v[:4]}" if len(v) < 6 else ""))
print("\n5. Session start dates (EDF header, dd.mm.yy) and times")
ses = defaultdict(set)
for r in rows:
    ses[(r["sub"], r["ses"])].add((r["date"], r["time"]))
for k in sorted(ses):
    d = sorted(ses[k]); print(f"   {k[0]} s{k[1]}: {d[0][0]} {d[0][1]} .. {d[-1][1]}  ({len({x[0] for x in d})} distinct dates)")
