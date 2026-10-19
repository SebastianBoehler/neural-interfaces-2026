"""Download and window organizer-recommended external data like PROTEUS.

Usage: python -I scripts/export_external.py {scherer2015,dreyer2023}
           [--subjects 1-20] [--no-download]

Raw files go to data/<name>/raw/ (gitignored), windows to
data/<name>/windows.npz with the keys of data/proteus_windows.npz plus
``interpolated`` (channels filled by spherical splines).

Preprocessing mirrors the official NeuralBench extractor: notch at every
50 and 60 Hz harmonic below min(Nyquist, 301 Hz), zero-phase FIR band-pass
0.1-75 Hz (low-pass dropped at or above Nyquist), resampling to 120 Hz,
volts -> microvolts, and one 4 s window starting 1 s after each cue. The
41 PROTEUS channels are then taken from the recording or interpolated from
all recorded EEG channels (MNE spherical splines, standard_1005 positions).
Labels: Scherer 2015 HAND -> 0, SUB -> 1, WORD -> 2 (NAV and FEET dropped);
Dreyer 2023 left/right-hand MI -> 0.
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SFREQ, START, DUR = 120.0, 1.0, 4.0
SCHERER_URL = "https://bnci-horizon-2020.eu/database/data-sets/004-2015/{}.mat"
SCHERER_IDS = "ACDEFGHJL"  # BNCI file names of the nine participants
SCHERER_CH = ("AFz F7 F3 Fz F4 F8 FC3 FCz FC4 T7 C3 Cz C4 T8 CP3 CPz CP4 P7 P5 "
              "P3 P1 Pz P2 P4 P6 P8 PO3 PO4 O1 O2").split()  # T3/T4 -> T7/T8
SCHERER_MAP = {4: 0, 2: 1, 1: 2}  # HAND, SUB, WORD (description.pdf codes)
SCHERER_CUE = 3.0  # cue 3 s after the coded trial start (description Fig. 1)
DREYER_MANIFEST = "https://osf.io/download/p5av2/"
DREYER_API = "https://files.de-1.osf.io/v1/resources/8tdk5/providers/osfstorage/"


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def preprocess(raw):
    sf = raw.info["sfreq"]
    notch = [f for base in (50.0, 60.0)
             for f in np.arange(base, min(sf / 2, 301), base)]
    raw.notch_filter(notch, picks="all", phase="zero", filter_length="auto",
                     verbose=False)
    raw.filter(0.1, 75.0 if 75.0 < sf / 2 else None, picks="all", verbose=False)
    return raw.resample(SFREQ, verbose=False)


def windows(raw, cues):
    """(n, C, 480) microvolt windows starting START s after each cue (s)."""
    X = raw.get_data() * 1e6
    n = int(round(DUR * SFREQ))
    starts = np.round((np.asarray(cues) + START) * SFREQ).astype(int)
    keep = (starts >= 0) & (starts + n <= X.shape[1])
    return np.stack([X[:, s:s + n] for s in starts[keep]]), keep


def to_proteus(W, ch_names, targets):
    """Map windows onto the 41 PROTEUS channels; returns (W41, interpolated)."""
    missing = [c for c in targets if c not in ch_names]
    names = list(ch_names) + missing
    info = mne.create_info(names, SFREQ, "eeg")
    data = np.concatenate([W, np.zeros((len(W), len(missing), W.shape[2]))], 1)
    ep = mne.EpochsArray(data, info, verbose=False)
    ep.set_montage("standard_1005")
    ep.info["bads"] = missing
    ep.interpolate_bads(reset_bads=True, verbose=False)
    return ep.get_data(picks=list(targets)).astype(np.float32), missing


def scherer(targets, subjects, download):
    raw_dir = ROOT / "data/scherer2015/raw"
    for i in subjects:
        sid = SCHERER_IDS[i - 1]
        path = raw_dir / f"{sid}.mat"
        if download:
            fetch(SCHERER_URL.format(sid), path)
        cells = loadmat(path, simplify_cells=True)["data"]
        for sess in cells:
            info = mne.create_info(SCHERER_CH, float(sess["fs"]), "eeg")
            raw = preprocess(mne.io.RawArray(sess["X"].T * 1e-6, info, verbose=False))
            y = np.asarray(sess["y"]).astype(int)
            cues = np.asarray(sess["trial"]) / float(sess["fs"]) + SCHERER_CUE
            sel = np.isin(y, list(SCHERER_MAP))
            W, keep = windows(raw, cues[sel])
            yield (f"Scherer2015/{sid}", str(sess["session"]), W, SCHERER_CH,
                   np.array([SCHERER_MAP[c] for c in y[sel][keep]]), cues[sel][keep])


def dreyer(targets, subjects, download):
    from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
    root = ROOT / "data/dreyer2023/raw"
    man = fetch(DREYER_MANIFEST, root / "manifest.tsv").read_text().splitlines()
    urls = dict(line.split("\t") for line in man[1:])
    for name in ("dataset_description.json", "participants.tsv"):
        fetch(DREYER_API + urls[name].rstrip("/").split("/")[-1], root / name)
    for i in subjects:
        sub = f"sub-{i:02d}"
        if download and not (root / sub).is_dir():
            z = fetch(DREYER_API + urls[f"{sub}.zip"].rstrip("/").split("/")[-1],
                      root / f"{sub}.zip")
            with zipfile.ZipFile(z) as f:
                assert all(not Path(n).is_absolute() and ".." not in Path(n).parts
                           for n in f.namelist())
                f.extractall(root)
            z.unlink()
        tasks = [t for t in get_entity_vals(root / sub, "task")
                 if "baseline" not in t and "rest" not in t
                 and not (i == 59 and t[:2] in ("R5", "R6"))]  # as in MOABB
        Ws, ys, cs, ch = [], [], [], None
        for t in sorted(tasks):
            raw = read_raw_bids(BIDSPath(subject=f"{i:02d}", task=t, suffix="eeg",
                                         root=root), verbose=False).load_data()
            raw.pick([c for c, k in zip(raw.ch_names, raw.get_channel_types())
                      if k == "eeg" and "EOG" not in c and "EMG" not in c])
            ev = [a["onset"] for a in raw.annotations if a["description"] in ("769", "770")]
            W, keep = windows(preprocess(raw), ev)
            assert ch is None or ch == raw.ch_names
            ch = raw.ch_names
            Ws.append(W), ys.append(np.zeros(len(W), int)), cs.append(np.asarray(ev)[keep])
        yield f"Dreyer2023/{sub}", "1", np.concatenate(Ws), ch, np.concatenate(ys), \
            np.concatenate(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=("scherer2015", "dreyer2023"))
    ap.add_argument("--subjects", default=None, help="e.g. 1-9")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    with np.load(ROOT / "data/proteus_windows.npz") as f:
        targets = [str(c) for c in f["ch_names"]]
    n_max = 9 if args.name == "scherer2015" else 87
    lo, _, hi = (args.subjects or f"1-{n_max}").partition("-")
    subjects = range(int(lo), int(hi or lo) + 1)
    reader = scherer if args.name == "scherer2015" else dreyer
    cols = {k: [] for k in ("X", "y", "subject", "session", "run", "onset")}
    interp = set()
    for subj, sess, W, ch, y, cues in reader(targets, subjects, not args.no_download):
        W41, missing = to_proteus(W, ch, targets)
        interp.update(missing)
        n = len(y)
        for k, v in (("X", W41), ("y", y), ("subject", [subj] * n),
                     ("session", [sess] * n), ("run", np.zeros(n, int)), ("onset", cues)):
            cols[k].append(np.asarray(v))
        print(subj, sess, W41.shape, np.bincount(y, minlength=3), flush=True)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(out["y"])
    out.update(split=np.full(n, "external"), interface=np.full(n, args.name),
               task=np.full(n, args.name), ch_names=np.asarray(targets),
               interpolated=np.asarray(sorted(interp)))
    assert out["X"].shape[1:] == (41, 480) and np.isfinite(out["X"]).all()
    np.savez(ROOT / "data" / args.name / "windows.npz", **out)
    print(json.dumps({"windows": n, "classes": np.bincount(out["y"]).tolist(),
                      "interpolated": sorted(interp)}))


if __name__ == "__main__":
    main()
