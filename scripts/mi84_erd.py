"""Where does the task start relative to the event marker? Mu (8-13 Hz) power at C3 / Cz / C4 in 1 s bins from -2 to +8 s, per class,
as the median over trials of 10 log10(power / power in the -2..0 s bins) (ERD < 0 dB), for Graz runs of one subject. python -I scripts/mi84_erd.py SUBJECT [N_RUNS]  (needs staged raw files)."""
import sys
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
import export_mi84 as E  # noqa: E402
from export_external import preprocess  # noqa: E402

sub, nrun = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4
acc, names = {}, None
for s, ses, run, base, stem, ext, want in E.recordings("graz"):
    if s != sub or len(acc.get("_n", [])) >= nrun:
        continue
    files = [E.fetch(f"{base}/{stem}{suf}", E.RAW / "ds007221" / f"{stem}{suf}") for suf in ("_eeg.eeg", "_eeg.vhdr", "_eeg.vmrk", "_events.tsv")]
    ev = E.read_tsv(files[-1])
    raw = preprocess(E.scalp(mne.io.read_raw_brainvision(files[1], preload=True, verbose=False)))
    ch = [raw.ch_names.index(c) for c in ("C3", "Cz", "C4")]
    b, a = butter(4, [8, 13], btype="band", fs=raw.info["sfreq"])
    p = np.abs(hilbert(filtfilt(b, a, raw.get_data()[ch]), axis=1)) ** 2
    for e in ev:
        t = int(round(float(e["onset"]) * raw.info["sfreq"]))
        if t - 240 < 0 or t + 960 > p.shape[1]:
            continue
        seg = p[:, t - 240:t + 960].reshape(3, 10, 120).mean(2)  # 10 bins, -2..+8 s
        acc.setdefault(e["trial_type"], []).append(10 * np.log10(seg / seg[:, :2].mean(1, keepdims=True)))
    acc.setdefault("_n", []).append(stem)
for c, v in sorted(acc.items()):
    if c != "_n":
        m = np.median(v, 0)
        print(f"{c:10s} n={len(v):3d}", *(f"{n}:" + " ".join(f"{x:+4.1f}" for x in m[i]) for i, n in enumerate(("C3", "Cz", "C4"))), sep="\n  ")
print("bins: -2..-1 -1..0 0..1 1..2 2..3 3..4 4..5 5..6 6..7 7..8 s relative to the marker; runs", acc["_n"])
