"""CFC (11 Nov, ledger s26): a tangent-space core on cross-frequency (phase-amplitude) coupling.

For each (slow band, fast band) pair the window is represented by 82 signals: the 41 slow-band signals and, for the same
channels, the log-envelope of the fast band filtered into the slow band. Their joint covariance holds, in its
off-diagonal block, how the slow rhythm at one channel co-varies with the fast-band amplitude at every channel
(phase-amplitude coupling across space); core1's machinery follows (CAR, trace shrinkage, causal log-Euclidean
recentring, tangent vectors, slow block, logistic head).
Usage: NI26_ROOT=... python cfc_core.py OUT_DIR
"""
import os, sys
ROOT = os.environ.get("NI26_ROOT", "/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [ROOT + "/experiments"]
import numpy as np
from scipy.signal import hilbert
import protocol
import ts_core2 as T

PAIRS = (((4, 8), (13, 30)), ((4, 8), (30, 45)), ((8, 13), (30, 45)), ((2, 6), (8, 13)))


def blocks_cfc(X, cfg):
    out = []
    bp = lambda Z, lo, hi: np.concatenate([T.bandpass(Z[i:i + 256], lo, hi) for i in range(0, len(Z), 256)])  # noqa: E731
    for (lo, hi), (flo, fhi) in PAIRS:
        S = bp(X, lo, hi); E = np.log(np.abs(hilbert(bp(X, flo, fhi), axis=-1)) + 1e-3).astype(np.float32); Ef = bp(E, lo, hi)
        S = S / np.sqrt((S.astype(np.float64) ** 2).mean()); Ef = Ef / np.sqrt((Ef.astype(np.float64) ** 2).mean())
        out.append(T.estimate(np.concatenate([S, Ef], 1).astype(np.float32), cfg["est"]))
    return out


T.blocks = blocks_cfc
if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); d = protocol.load()
    T.run(d, "car_fb7_full_t0.1_le_k2", ["lr3", "lr10"], out)
