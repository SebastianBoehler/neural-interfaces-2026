"""Signal-quality profile per public session: is the hard person (P04) hard because of the recording?"""
import sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/experiments")
import protocol
d = protocol.load(); pro = d["split"] != "external"
X, st, y = d["X"], d["stream"], d["y"]
print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in d.items()})
print("session  n | amp uV-ish med p99/med | chan-var spread (log10 max/med, min/med) | lowcorr chans | hf(30-55)/alpha | 50Hz ratio | delta(<2Hz) share | win outlier %")
f = np.fft.rfftfreq(480, 1 / 120)
for s in dict.fromkeys(st[pro]):
    r = np.flatnonzero((st == s) & pro); x = X[r].astype(np.float64); x = x - x.mean(2, keepdims=True)
    v = x.var(2); cv = np.median(v, 0)  # per-channel variance (median over windows)
    amp = np.sqrt(np.median(v)); p99 = np.sqrt(np.percentile(v, 99))
    xc = x - x.mean(1, keepdims=True)
    C = np.mean([np.corrcoef(w) for w in xc[:: max(1, len(xc) // 40)]], 0); np.fill_diagonal(C, np.nan)
    low = int((np.nanmax(np.abs(C), 1) < 0.4).sum())
    P = (np.abs(np.fft.rfft(x * np.hanning(480), axis=2)) ** 2).mean((0, 1))
    band = lambda a, b: P[(f >= a) & (f < b)].mean()  # noqa: E731
    wv = np.log(v.mean(1)); out = np.mean(np.abs(wv - np.median(wv)) > 3 * 1.4826 * np.median(np.abs(wv - np.median(wv))))
    print(f"{s:28s} {len(r):4d} | {amp:8.3g} {p99 / amp:5.1f} | {np.log10(cv.max() / np.median(cv)):5.2f} {np.log10(cv.min() / np.median(cv)):6.2f} | {low:2d} | {band(30, 55) / band(8, 13):7.4f} | {band(49, 51) / band(44, 48):6.2f} | {P[(f < 2)].sum() / P.sum():5.2f} | {100 * out:5.1f}")
