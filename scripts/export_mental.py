"""Window public mental-arithmetic EEG corpora like PROTEUS, for pretraining.

Usage: python -I scripts/export_mental.py {zyma2019,cmx7mm} [--subjects 1-30]
           [--out data/<name>/windows_<lo>-<hi>.npz]

Raw files (gitignored) are read from data/<name>_raw/:
- zyma2019: PhysioNet eegmat 1.0.0 (Zyma et al. 2019, ODC-By 1.0,
  https://physionet.org/content/eegmat/1.0.0/), subjects 0-35.
  SubjectNN_1.edf is rest before the task (~180 s), SubjectNN_2.edf serial
  subtraction (~60 s); 19 10-20 channels at 500 Hz (T3/T4/T5/T6 renamed
  T7/T8/P7/P8), already filtered and ICA-cleaned by the authors.
- cmx7mm: OpenNeuro ds007554 1.0.0 (CMx7-MM, Ajra et al. 2026, CC0,
  arXiv:2603.22933), subjects 1-30, BIDS EDF, 32 channels at 250 Hz,
  3 sessions; tasks mentalarithmetic, nback and motorimagery (right arm).
  sub-010..014 are skipped: the data paper reports their EEG unusable;
  so are saturated recordings (median channel std above 1 mV after
  filtering: all of sub-009, ~7.4 mV on every channel like sub-010..013).

Preprocessing, channel mapping and labels as export_moabb.py (official
chain; spherical splines onto the 41 PROTEUS channels; sorted dataset-local
class names, one head per corpus in experiments/pretrain.py). Blocks give
consecutive non-overlapping 4 s windows, the first 1 s after block onset:
zyma2019 whole files; cmx7mm arithmetic and n-back from the "start" marker
(25 s) to the last stimulus offset (205 s), and "pretask" from recording
onset to the first event (25 s) of every recording (undocumented; the paper
only says tasks were separated by 30 s rest). Motor imagery: one window 0-4 s
after each beep, except the first row, which opens the task at 25 s like
"start" (files list 19 cues, the paper 18). Channels whose windowed std
exceeds 5x the recording median (or is flat) are dropped, then interpolated
when they are PROTEUS channels. ``onset`` is the window start (s).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_external import DUR, SFREQ, preprocess, to_proteus  # noqa: E402

CLASSES = {"zyma2019": ["rest", "subtraction"],
           "cmx7mm": ["arithmetic", "motor_imagery", "nback", "pretask"]}
SUBJECTS = {"zyma2019": "0-35", "cmx7mm": "1-30"}
ZYMA_MAP = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
CMX_TASKS = {"mentalarithmetic": 0, "motorimagery": 1, "nback": 2}
CMX_SKIP = range(10, 15)  # arXiv:2603.22933: EEG unusable (device problems)


def block(lo, hi):
    """Starts (s) of consecutive 4 s windows in [lo + 1, hi]."""
    return lo + 1.0 + DUR * np.arange(max(0, int((hi - lo - 1.0) // DUR)))


def cut(raw, starts, y):
    """(n, C, 480) microvolt windows at starts (s), outlier channels dropped."""
    X, n = preprocess(raw).get_data() * 1e6, int(DUR * SFREQ)
    starts, y = np.asarray(starts, float), np.asarray(y)
    st = np.round(starts * SFREQ).astype(int)
    ok = (st >= 0) & (st + n <= X.shape[1])
    W = np.stack([X[:, a:a + n] for a in st[ok]])
    sd = W.std((0, 2))
    good = (sd < 5 * np.median(sd)) & (sd > 0.1)
    if not good.all():
        print("  dropped", [c for c, g in zip(raw.ch_names, good) if not g], flush=True)
    return W[:, good], [c for c, g in zip(raw.ch_names, good) if g], y[ok], starts[ok]


def zyma2019(subjects):
    root = ROOT / "data/zyma2019_raw"
    for s in subjects:
        for k in (1, 2):  # _1 rest before the task, _2 serial subtraction
            raw = mne.io.read_raw_edf(root / f"Subject{s:02d}_{k}.edf", preload=True, verbose=False)
            raw.pick([c for c in raw.ch_names if c.startswith("EEG ") and "-" not in c])
            raw.rename_channels(lambda c: ZYMA_MAP.get(c[4:], c[4:]))
            st = block(0.0, raw.n_times / raw.info["sfreq"])
            W, ch, y, on = cut(raw, st, np.full(len(st), k - 1))
            yield f"Zyma2019/{s}", "0", str(k), W, ch, y, on


def cmx7mm(subjects):
    root = ROOT / "data/cmx7mm_raw"
    for s in subjects:
        if s in CMX_SKIP:
            print(f"skip cmx7mm/{s}: EEG unusable per data paper", flush=True)
            continue
        for f in sorted(root.glob(f"sub-{s:03d}/ses-*/eeg/*_eeg.edf")):
            task = f.name.split("task-")[1].split("_")[0]
            if task not in CMX_TASKS:
                continue
            with open(str(f).replace("_eeg.edf", "_events.tsv")) as h:
                ev = list(csv.DictReader(h, delimiter="\t"))
            on = np.array([float(e["onset"]) for e in ev])
            if task == "motorimagery":
                task_st = on[1:]  # first row opens the task (see docstring)
            else:
                assert ev[0]["trial_type"] == "start", f
                task_st = block(on[0], on[-1] + float(ev[-1]["duration"]))
            pre = block(0.0, on[0])
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
            raw.pick([c for c in raw.ch_names if c != "Status"])
            raw.rename_channels({"FP1": "Fp1", "FP2": "Fp2"})
            y = np.r_[np.full(len(pre), 3), np.full(len(task_st), CMX_TASKS[task])]
            W, ch, y, st = cut(raw, np.r_[pre, task_st], y)
            if np.median(W.std((0, 2))) > 1000:
                print(f"skip {f.name}: saturated", flush=True)
                continue
            yield f"CMx7MM/{s}", f.parts[-3], task, W, ch, y, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(CLASSES))
    ap.add_argument("--subjects", default=None, help="e.g. 1-30 (default: all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with np.load(ROOT / "data/proteus_windows.npz") as f:
        targets = [str(c) for c in f["ch_names"]]
    lo, _, hi = (args.subjects or SUBJECTS[args.name]).partition("-")
    out = Path(args.out or ROOT / "data" / args.name / f"windows_{lo}-{hi or lo}.npz")
    cols = {k: [] for k in ("X", "y", "subject", "session", "run", "onset")}
    interp, classes = set(), CLASSES[args.name]
    reader = zyma2019 if args.name == "zyma2019" else cmx7mm
    for subj, sess, run, W, ch, y, starts in reader(range(int(lo), int(hi or lo) + 1)):
        W41, missing = to_proteus(W, ch, targets)
        interp.update(missing)
        n = len(y)
        for k, v in (("X", W41), ("y", y), ("subject", [subj] * n), ("session", [sess] * n),
                     ("run", [run] * n), ("onset", starts)):
            cols[k].append(np.asarray(v))
        print(subj, sess, run, W41.shape, np.bincount(y, minlength=len(classes)).tolist(), flush=True)
    d = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(d["y"])
    d.update(split=np.full(n, "external"), interface=np.full(n, args.name),
             task=np.full(n, args.name), ch_names=np.asarray(targets),
             classes=np.asarray(classes), interpolated=np.asarray(sorted(interp)))
    assert d["X"].shape[1:] == (41, 480) and np.isfinite(d["X"]).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    print(json.dumps({"out": str(out), "windows": n, "classes": list(classes),
                      "counts": np.bincount(d["y"]).tolist(), "interpolated": len(interp)}))


if __name__ == "__main__":
    main()
