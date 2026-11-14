"""CAUSAL PERSONALISATION features (ledger s253): personal_ceiling/classic.py's features WITHOUT the whole-session alignment
(s246's EA used the session mean covariance, which is not causal). Everything is per window: zero-phase band filtering
inside the 4 s window, OAS band covariances for classic.TSB, classic.CSPB and 8-30 Hz, < 4 Hz waveform every 15th sample.
Same keys as classic.features, so classic.fit_predict runs on it unchanged. Cached in runs/causalpers/feat.npz.
Usage: python studies/14nov/causalpers/feats.py"""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO / "studies/14nov/personal_ceiling"))
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
import classic as CL  # noqa: E402

OUT = REPO / "runs/causalpers/feat.npz"; BANDS = sorted(set(CL.TSB + CL.CSPB + [(8, 30)]))


def window_feats(X, bands=BANDS):
    """X (n, 41, 480) -> {key: per-window features}; no state across windows"""
    out = {}
    for lo, hi in bands:
        Xb = sosfiltfilt(butter(4, (lo, hi), "bandpass", fs=CL.FS, output="sos"), X, axis=-1); out[f"cov{lo}_{hi}"] = Covariances("oas").fit_transform(Xb)
    out["slow"] = sosfiltfilt(butter(4, 4.0, "lowpass", fs=CL.FS, output="sos"), X, axis=-1)[:, :, ::15].reshape(len(X), -1)
    return out


def load():
    if OUT.exists():
        z = np.load(OUT); return {k: z[k] for k in z.files}
    t0 = time.time()
    with np.load(REPO / "data/proteus_windows.npz") as w:
        X = w["X"].astype(np.float64)
    out = window_feats(X); np.savez(OUT, **out); print(f"features {len(X)} windows {time.time() - t0:.0f}s -> {OUT}", flush=True); return out


if __name__ == "__main__":
    load()
