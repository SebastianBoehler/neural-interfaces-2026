"""AEC (11 Nov, ledger s25): a tangent-space core on amplitude-envelope covariances, a kind of feature no member uses.

Per band: band-pass, Hilbert envelope, log; the 41 x 41 covariance of the log-envelopes over the window (how channel
amplitudes co-fluctuate), then exactly core1's machinery (CAR, trace shrinkage 0.1, causal log-Euclidean recentring k 2,
tangent vectors, slow block, logistic head). Bands: fb7. Heads lr3 / lr10. Writes OUT/<name>.json and .npz
(probabilities for all_loso and seen_session, as ts_core2.run).
Usage: NI26_ROOT=... python aec_core.py OUT_DIR
"""
import os, sys
ROOT = os.environ.get("NI26_ROOT", "/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [ROOT + "/experiments"]
import numpy as np
from scipy.signal import hilbert
import protocol
import ts_core2 as T


def blocks_env(X, cfg):
    out = []
    for lo, hi in T.bandset(cfg["bands"]):
        Xb = np.concatenate([T.bandpass(X[i:i + 256], lo, hi) for i in range(0, len(X), 256)])
        E = np.log(np.abs(hilbert(Xb, axis=-1)) + 1e-3).astype(np.float32)
        out.append(T.estimate(E, cfg["est"]))
    return out


T.blocks = blocks_env
if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); d = protocol.load()
    T.run(d, "car_fb7_full_t0.1_le_k2", ["lr3", "lr10"], out)
