"""s278 step 1: Shin 2018 word generation (WG) vs baseline windows, PROTEUS layout.

Usage: python -I studies/15nov/shin278/export_shin.py [--subjects 1-26] [--out PATH]

Raw BrainVision zips: data/shin2018/raw/VPnnn.zip (DepositOnce 10.14279/depositonce-5830).
Markers S 16 / S 32 open the two 10 s conditions (WG / baseline, see scripts/export_moabb.py);
two 4 s windows per trial, starting 1 s and 5 s after the marker. Preprocessing and the
41-channel mapping are scripts/export_external.py (notch, 0.1-75 Hz, 120 Hz, microvolts).
Keys as data/<corpus>/windows.npz plus trial (within-run trial index), block (= run 1-3,
the block id of the ledger), win (0/1 window within the trial), recorded (channel
intersect with the 41 PROTEUS channels), interpolated (the rest, spline-filled).
Classes: 0 = baseline, 1 = WG. Asserts: 26 people, 3 runs each, 20 trials per run,
10 per class, 2 windows per trial (so 30 trials and 60 windows per class and person).
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from export_external import SFREQ, preprocess, to_proteus  # noqa: E402

KEEP = ["Stimulus/S 32", "Stimulus/S 16"]  # baseline -> 0, word generation -> 1
BLOCK_OFFSETS = (0.0, 4.0)  # windows at marker+1 s and +5 s (stimulus lasts 10 s)
DUR = 4.0


def header(d, run):
    """vhdr path; VP017 run 2 names wor2.eeg/wor2.vmrk in its headers (dataset typo): fixed copy."""
    v = d / f"word{run}.vhdr"
    t = v.read_text(errors="replace")
    if f"=wor{run}." not in t:
        return v
    fixed = d / f"word{run}_fixed.vhdr"
    fixed.write_text(t.replace(f"wor{run}.", f"word{run}."))
    m = d / f"word{run}_fixed.vmrk"
    m.write_text((d / f"word{run}.vmrk").read_text(errors="replace").replace(f"wor{run}.", f"word{run}."))
    fixed.write_text(fixed.read_text().replace(f"word{run}.vmrk", m.name))
    print("repaired header", fixed, flush=True)
    return fixed


def person(s, targets):
    d = ROOT / f"data/shin2018/raw/VP{s:03d}"
    if not (d / "word1.vhdr").is_file():
        with zipfile.ZipFile(d.with_suffix(".zip")) as f:
            assert all(not Path(n).is_absolute() and ".." not in Path(n).parts
                       for n in f.namelist())
            f.extractall(d)
    pos = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    for run in (1, 2, 3):
        raw = mne.io.read_raw_brainvision(header(d, run), preload=True, verbose=False)
        raw.pick([c for c in raw.ch_names if c in pos])
        a = [(o, KEEP.index(t)) for o, t in zip(raw.annotations.onset, raw.annotations.description)
             if t in KEEP]
        assert len(a) == 20 and sum(c for _, c in a) == 10, (s, run, len(a))
        X = preprocess(raw).get_data() * 1e6
        n = int(DUR * SFREQ)
        rows = [(o + k, c, i, w) for i, (o, c) in enumerate(a) for w, k in enumerate(BLOCK_OFFSETS)]
        st = np.round(np.array([r[0] + 1.0 for r in rows]) * SFREQ).astype(int)
        assert (st + n <= X.shape[1]).all(), (s, run, "window beyond recording")
        W = np.stack([X[:, t:t + n] for t in st])
        W41, interp = to_proteus(W, raw.ch_names, targets)
        yield run, W41, raw.ch_names, interp, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-26")
    ap.add_argument("--out", default=str(ROOT / "data/shin2018/windows.npz"))
    args = ap.parse_args()
    with np.load(ROOT / "data/proteus_windows.npz") as f:
        targets = [str(c) for c in f["ch_names"]]
    lo, _, hi = args.subjects.partition("-")
    cols = {k: [] for k in ("X", "y", "subject", "session", "run", "onset", "trial", "block", "win")}
    interp_all, rec_all = set(), set()
    for s in range(int(lo), int(hi or lo) + 1):
        for run, W41, ch, interp, rows in person(s, targets):
            interp_all.update(interp)
            rec_all.update(c for c in ch if c in targets)
            n = len(rows)
            for k, v in (("X", W41), ("y", [r[1] for r in rows]), ("subject", [f"Shin2018/{s}"] * n),
                         ("session", ["0"] * n), ("run", [str(run)] * n),
                         ("onset", [r[0] for r in rows]), ("trial", [r[2] for r in rows]),
                         ("block", [run] * n), ("win", [r[3] for r in rows])):
                cols[k].append(np.asarray(v))
            print(s, run, W41.shape, flush=True)
    d = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(d["y"])
    if (lo, hi) == ("1", "26"):  # full-corpus asserts
        subj = d["subject"]
        assert len(set(subj)) == 26, len(set(subj))
        for p in set(subj):
            m = subj == p
            assert sorted(set(d["block"][m])) == [1, 2, 3]
            assert np.bincount(d["y"][m]).tolist() == [60, 60], (p, np.bincount(d["y"][m]))
            assert len({(b, t) for b, t in zip(d["block"][m], d["trial"][m])}) == 60
    rec = [c for c in targets if c in rec_all]
    d.update(split=np.full(n, "external"), interface=np.full(n, "Shin2018WG"),
             task=np.full(n, "Shin2018WG"), ch_names=np.asarray(targets),
             classes=np.asarray(["baseline", "WG"]), recorded=np.asarray(rec),
             interpolated=np.asarray(sorted(interp_all)))
    assert d["X"].shape[1:] == (41, 480) and np.isfinite(d["X"]).all()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **d)
    print(json.dumps({"out": args.out, "windows": n, "people": len(set(d["subject"])),
                      "counts": np.bincount(d["y"]).tolist(), "recorded": len(rec),
                      "interpolated": sorted(interp_all)}))


if __name__ == "__main__":
    main()
