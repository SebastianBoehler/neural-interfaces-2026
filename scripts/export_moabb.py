"""Window MOABB motor-imagery corpora like PROTEUS, for pretraining (R10).

Usage: python -I scripts/export_moabb.py DATASET [--subjects 1-20]
           [--out data/<dataset>/windows_<lo>-<hi>.npz]

Raw files are cached by MOABB under $MNE_DATA (gitignored data/moabb_raw).
Preprocessing is export_external.preprocess (official NeuralBench chain:
notches, 0.1-75 Hz, 120 Hz, microvolts); channels are mapped onto the 41
PROTEUS channels (spherical splines for missing ones). One 4 s window per
cue, starting max(0, min(1, tmax - 4)) s after it so that it covers the
dataset's task interval; trials of 8 s or more give consecutive 4 s windows. Labels are dataset-local (0..K-1, sorted event
names) and kept apart from the PROTEUS classes: pretraining uses one head
per dataset (experiments/pretrain.py). Only openly licensed corpora:
see docs/OPTIMIZATION_R10.md.
"""

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_external import DUR, SFREQ, preprocess, to_proteus  # noqa: E402

# dataset -> events kept (None: all of the dataset's event_id)
KEEP = {"Cho2017": None, "Lee2019_MI": None, "PhysionetMI": None,
        "Stieger2021": None, "Dreyer2023": None, "Schirrmeister2017": None,
        "Shin2017A": None, "Shin2017B": None, "Liu2024": None, "Yang2025": None,
        # wave 10m (3 Nov; licences on the data hosts: CC BY 4.0 or CC0, healthy participants)
        "Zhou2020": None, "Chang2025": None, "GuttmannFlury2025_MI": None, "Kumar2024": None,
        "Rozado2015": None, "Weibo2014": ["feet", "hands", "left_hand", "rest", "right_hand"]}
# Shin 2017: MOABB asks to accept the GNU GPL data terms (user OK, 28 Oct); Stieger 2021: sessions 1-3
# of up to 11 (377 GB in all)
ARGS = {"Shin2017A": {"accept": True}, "Shin2017B": {"accept": True}, "Stieger2021": {"sessions": [1, 2, 3]}}


def load(name, subjects):
    import moabb.datasets as md
    ds = getattr(md, name)(**ARGS.get(name, {}))
    keep = sorted(KEEP[name] or ds.event_id)
    lab = {ds.event_id[e]: i for i, e in enumerate(keep)}
    t0, tlen = float(ds.interval[0]), float(ds.interval[1]) - float(ds.interval[0])  # t0 = 0 up to R10
    start = max(0.0, min(1.0, tlen - DUR))
    offs = [start] + [start + k * DUR for k in range(1, 9) if start + (k + 1) * DUR <= tlen]
    for s in subjects:
        try:
            data = ds.get_data([s])[s]
        except Exception as e:  # e.g. an unreadable Dreyer 2023 recording: skip, logged
            print(f"skip {name}/{s}: {e!r}", flush=True)
            continue
        for sess, runs in data.items():
            for run, raw in runs.items():
                raw = raw.copy().load_data()
                ev, _ = mne.events_from_annotations(raw, event_id={e: ds.event_id[e] for e in keep},
                                                    verbose=False)
                if not len(ev):
                    continue
                raw.pick("eeg", exclude="bads")
                ev = np.concatenate([ev] * len(offs))  # long trials: consecutive 4 s windows
                cues = ev[:, 0] / raw.info["sfreq"] + t0 + np.repeat(
                    np.array(offs) - start, len(ev) // len(offs))
                X = preprocess(raw).get_data() * 1e6
                n, st = int(DUR * SFREQ), np.round((cues + start) * SFREQ).astype(int)
                ok = (st >= 0) & (st + n <= X.shape[1])
                if ok.sum():
                    W = np.stack([X[:, a:a + n] for a in st[ok]])
                    yield (f"{name}/{s}", str(sess), str(run), W, raw.ch_names,
                           np.array([lab[c] for c in ev[ok, 2]]), cues[ok]), keep


def shin2018(name, subjects):
    """Shin et al. 2018 word generation (DepositOnce 10.14279/depositonce-5830,
    CC0), BrainVision files data/shin2018/raw/VPnnn.zip: markers S 16 / S 32
    open the two 10 s conditions (word generation, baseline); two windows per
    trial, 1-5 s and 5-9 s after the marker."""
    import zipfile
    root, pos = ROOT / "data/shin2018/raw", set(mne.channels.make_standard_montage("standard_1005").ch_names)
    keep = ["Stimulus/S 16", "Stimulus/S 32"]
    for s in subjects:
        d = root / f"VP{s:03d}"
        if not d.is_dir():
            with zipfile.ZipFile(root / f"VP{s:03d}.zip") as f:
                assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in f.namelist())
                f.extractall(d)  # flat archives (VP002 on): one folder per participant
        for run in (1, 2, 3):
            raw = mne.io.read_raw_brainvision(d / f"word{run}.vhdr", preload=True, verbose=False)
            raw.pick([c for c in raw.ch_names if c in pos])
            a = [(o, keep.index(t)) for o, t in zip(raw.annotations.onset, raw.annotations.description) if t in keep]
            cues = np.array([o + k for o, _ in a for k in (0.0, 4.0)])
            y = np.array([c for _, c in a for _ in (0, 1)])
            X, n = preprocess(raw).get_data() * 1e6, int(DUR * SFREQ)
            st = np.round((cues + 1.0) * SFREQ).astype(int)
            ok = st + n <= X.shape[1]
            yield (f"{name}/{s}", "0", str(run), np.stack([X[:, a:a + n] for a in st[ok]]),
                   raw.ch_names, y[ok], cues[ok]), ["S16", "S32"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(KEEP) + ["Shin2018WG"])
    ap.add_argument("--subjects", required=True, help="e.g. 1-20")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with np.load(ROOT / "data/proteus_windows.npz") as f:
        targets = [str(c) for c in f["ch_names"]]
    lo, _, hi = args.subjects.partition("-")
    out = Path(args.out or ROOT / "data" / args.name.lower() / f"windows_{lo}-{hi or lo}.npz")
    cols = {k: [] for k in ("X", "y", "subject", "session", "run", "onset")}
    interp, classes = set(), None
    for (subj, sess, run, W, ch, y, cues), classes in (shin2018 if args.name == "Shin2018WG" else load)(
            args.name, range(int(lo), int(hi or lo) + 1)):
        W41, missing = to_proteus(W, ch, targets)
        interp.update(missing)
        n = len(y)
        for k, v in (("X", W41), ("y", y), ("subject", [subj] * n), ("session", [sess] * n),
                     ("run", [run] * n), ("onset", cues)):
            cols[k].append(np.asarray(v))
        print(subj, sess, run, W41.shape, np.bincount(y).tolist(), flush=True)
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
