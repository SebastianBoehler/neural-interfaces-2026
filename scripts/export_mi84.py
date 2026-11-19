"""Stage and window ds007221 (OpenNeuro v1.0.3, CC0) and NETBCI (NEMAR nm000305 v1.0.1, CC BY 4.0) like PROTEUS (lane B6).

python -I scripts/export_mi84.py {graz,hybrid,netbci} --subjects 1-3 [--max-runs N] [--start S]
  graz    ds007221 Graz group (sub-01..14): left/right hand, feet, rest      -> data/ds007221g/windows_<lo>-<hi>.npz
  hybrid  ds007221 Hybrid (sub-37..73) + Hybrid-online (sub-74..84) people, task acq-graz only (offline Graz paradigm; flicker, video and
          online-feedback conditions excluded)                                  -> data/ds007221lr/windows_<lo>-<hi>.npz
  netbci  NETBCI sub-1..19, right-hand MI vs rest (online cursor task)         -> data/netbci/windows_<lo>-<hi>.npz
--subjects indexes the participant numbers inside the group's participant_id (sub-NN / sub-N). Raw files: data/mi84_raw (gitignored, kept).
Preprocessing is export_external.preprocess (notch at 50/60 Hz harmonics, 0.1-75 Hz at native rate, 120 Hz, microvolts) and to_proteus
(our 41-channel order, spherical splines for missing), exactly as export_moabb.py; one 4 s window START s after every marker
(rule of export_moabb, start = max(0, min(1, task length - 4)): 0 s for ds007221, 1 s for NETBCI; see mi84_erd.py). Labels: explicit name -> local index map in experiments/class_map.OWN_HEAD (unmapped names raise). Checks per recording:
events.tsv value<->name pairing stays constant, trial count equals the recordings.tsv count, window fits the recording."""
import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments")]
import class_map  # noqa: E402
from export_external import DUR, SFREQ, preprocess, to_proteus  # noqa: E402

OPEN = "https://s3.amazonaws.com/openneuro.org/ds007221"
NEMAR = "https://data.nemar.org/nm000305/v1.0.1"
RAW = ROOT / "data/mi84_raw"
NONEEG = {"HEO", "VEO", "EKG", "EMG", "TRIGGER", "CB1", "CB2"}
MAXMISS = 6  # PROTEUS channels spline-filled per recording before the export refuses
CODES = {"graz": {"left_hand": 1, "right_hand": 2, "feet": 3, "rest": 4}}


def fetch(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            n = int(r.headers.get("Content-Length", -1))
            while b := r.read(1 << 22):
                f.write(b)
        assert n < 0 or tmp.stat().st_size == n, f"short download {url}"
        tmp.rename(path)
    return path


def read_tsv(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8-sig")), delimiter="\t"))


def scalp(raw):
    """Keep scalp electrodes under their standard_1005 spelling (files use FZ / IZ ...); drop EOG / ECG / EMG / trigger channels."""
    std = {c.upper(): c for c in mne.channels.make_standard_montage("standard_1005").ch_names}
    keep = [c for c in raw.ch_names if c.upper() in std and c.upper() not in NONEEG]
    raw.pick(keep)
    raw.rename_channels({c: std[c.upper()] for c in keep})
    return raw


def recordings(group):
    """Yield (subject_no, session, run, trial_names, raw loader, cue seconds list) per recording of the group."""
    if group == "netbci":
        for s in range(1, 20):
            for ses in range(1, 5):
                for run in range(1, 7):
                    stem = f"sub-{s}/ses-{ses:02d}/eeg/sub-{s}_ses-{ses:02d}_task-imagery_run-{run:02d}"
                    yield s, f"ses-{ses:02d}", f"run-{run:02d}", NEMAR, stem, ".edf", None
        return
    rec = read_tsv(fetch(f"{OPEN}/recordings.tsv", RAW / "ds007221/recordings.tsv"))
    for r in rec:
        graz = r["task"] == "graz" and group == "graz"
        hyb = r["task"] in ("hybrid", "hybridonline") and r["acquisition"] == "graz" and group == "hybrid"
        if graz or hyb:
            s = int(r["subject"][4:])
            sub = r["subject"]
            yield s, r["session"], f"run-{int(r['run']):02d}", OPEN, f"{sub}/{r['session']}/eeg/{r['file']}", ".vhdr", int(r["trial_count"])


def window_run(group, base, stem, ext, want, start, targets):
    files = [fetch(f"{base}/{stem}{suf}", RAW / ("ds007221" if base == OPEN else "netbci") / f"{stem}{suf}")
             for suf in (("_eeg.eeg", "_eeg.vhdr", "_eeg.vmrk") if ext == ".vhdr" else ("_eeg.edf",)) + ("_events.tsv",)]
    ev = read_tsv(files[-1])
    names = [e["trial_type"] for e in ev]
    pairs = {(e["trial_type"], e["value"]) for e in ev}
    assert len({v for _, v in pairs}) == len(pairs) == len({n for n, _ in pairs}), f"{stem}: name/value pairing ambiguous {pairs}"
    if group == "graz":
        assert all(CODES["graz"][n] == int(v) for n, v in pairs), f"{stem}: event codes differ from the pinned pairing {pairs}"
    assert want is None or len(ev) == want, f"{stem}: {len(ev)} events, recordings.tsv says {want}"
    y = class_map.own_head_labels({"graz": "ds007221g", "hybrid": "ds007221lr", "netbci": "netbci"}[group], names)
    raw = (mne.io.read_raw_brainvision(files[1], preload=True, verbose=False) if ext == ".vhdr"
           else mne.io.read_raw_edf(files[0], preload=True, verbose=False))
    assert abs(raw.times[-1] * raw.info["sfreq"] - raw.n_times) < raw.info["sfreq"]
    cues = np.array([float(e["onset"]) for e in ev])
    X = preprocess(scalp(raw)).get_data() * 1e6
    n, st = int(round(DUR * SFREQ)), np.round((cues + start) * SFREQ).astype(int)
    ok = (st >= 0) & (st + n <= X.shape[1])
    assert ok.mean() > 0.9, f"{stem}: {ok.sum()}/{len(ok)} windows fit"
    W = np.stack([X[:, a:a + n] for a in st[ok]])
    W41, missing = to_proteus(W, raw.ch_names, targets)
    assert len(missing) <= MAXMISS, f"{stem}: {len(missing)} spline-filled channels {missing}"
    return W41, y[ok], cues[ok], missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=("graz", "hybrid", "netbci"))
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--start", type=float, default=None, help="s after the marker; default 0 for ds007221 (marker = task onset, 4 s task), 1 for NETBCI (5 s target)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with np.load(ROOT / "data/proteus_windows.npz") as f:
        targets = [str(c) for c in f["ch_names"]]
    a.start = (1.0 if a.group == "netbci" else 0.0) if a.start is None else a.start
    lo, _, hi = a.subjects.partition("-")
    lo, hi = int(lo), int(hi or lo)
    corpus = {"graz": "ds007221g", "hybrid": "ds007221lr", "netbci": "netbci"}[a.group]
    out = Path(a.out or ROOT / "data" / corpus / f"windows_{lo:02d}-{hi:02d}.npz")
    cols = {k: [] for k in ("X", "y", "subject", "session", "run", "onset")}
    interp, done = set(), 0
    for s, ses, run, base, stem, ext, want in recordings(a.group):
        if not lo <= s <= hi or (a.max_runs and done >= a.max_runs):
            continue
        W, y, cues, missing = window_run(a.group, base, stem, ext, want, a.start, targets)
        interp.update(missing)
        done += 1
        for k, v in (("X", W), ("y", y), ("subject", [f"{corpus}/{s}"] * len(y)), ("session", [ses] * len(y)),
                     ("run", [run] * len(y)), ("onset", cues)):
            cols[k].append(np.asarray(v))
        print(corpus, s, ses, run, W.shape, np.bincount(y, minlength=len(class_map.OWN_HEAD[corpus])).tolist(), len(missing), flush=True)
    assert done, "no recording matched"
    d = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(d["y"])
    d.update(split=np.full(n, "external"), interface=np.full(n, corpus), task=np.full(n, corpus), ch_names=np.asarray(targets),
             classes=np.asarray(class_map.OWN_HEAD[corpus]), interpolated=np.asarray(sorted(interp)), cue_start_s=np.float32(a.start))
    assert d["X"].shape[1:] == (41, 480) and np.isfinite(d["X"]).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    print(json.dumps({"out": str(out), "runs": done, "windows": n, "classes": class_map.OWN_HEAD[corpus],
                      "counts": np.bincount(d["y"]).tolist(), "interpolated": sorted(interp)}))


if __name__ == "__main__":
    main()
