"""Performance panel, 14 Nov: isolated CPU micro-benchmarks of the pure numpy/torch/scipy operations the package
performs per window (synthetic data, no package code, no weights). Minimum over repeats; one BLAS thread as the package sets it."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi
torch.set_num_threads(1)
try:
    from threadpoolctl import threadpool_limits; _l = threadpool_limits(limits=1)
except ImportError:
    pass
rng = np.random.default_rng(0)


def best(f, n=7):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts)


def spd(b, n=41, cplx=False):
    A = rng.standard_normal((b, n, 120)) + (1j * rng.standard_normal((b, n, 120)) if cplx else 0)
    return A @ A.conj().transpose(0, 2, 1) / 120 + 1e-3 * np.eye(n)


def eigb(C):
    w, V = (t.numpy() for t in torch.linalg.eigh(torch.from_numpy(C)))
    return (V * np.log(w)[:, None, :]) @ V.conj().transpose(0, 2, 1)


C1, C2 = spd(51), spd(21, cplx=True)
print(f"eigh+rebuild 51x41x41 f64 (one of core1's three passes): {best(lambda: eigb(C1)):.2f} ms")
print(f"eigh+rebuild 21x41x41 c128 (one of core2's three passes): {best(lambda: eigb(C2)):.2f} ms")
print(f"  eigh only, real: {best(lambda: torch.linalg.eigh(torch.from_numpy(C1))):.2f} ms; complex: {best(lambda: torch.linalg.eigh(torch.from_numpy(C2))):.2f} ms")
for k in (2, 4, 6):
    pool = ThreadPoolExecutor(k)
    parts = np.array_split(np.arange(51), k); parts2 = np.array_split(np.arange(21), k)
    f = lambda: [list(pool.map(lambda i: eigb(C1[i]), parts)), list(pool.map(lambda i: eigb(C2[i]), parts2))]
    ref = np.concatenate(list(pool.map(lambda i: eigb(C1[i]), parts)))
    print(f"  both cores' pass, {k} threads over chunks: {best(f):.2f} ms (1 thread: {best(lambda: (eigb(C1), eigb(C2))):.2f}); bitwise equal to unchunked: {np.array_equal(ref, eigb(C1))}")
    pool.shutdown()
T1 = torch.from_numpy(C1)
print(f"torch.linalg.matrix_exp 51x41x41 f64 (alternative to pass 2): {best(lambda: torch.linalg.matrix_exp(-0.5 * T1)):.2f} ms")

# band filters: 17 bands, 4th-order band-pass (4 sections), filtfilt on (41, 480) with padding
x = rng.standard_normal((41, 480))
bank = [butter(4, (lo, lo + 4), btype="band", fs=120, output="sos") for lo in np.linspace(1, 40, 17)]
zis = [sosfilt_zi(s) for s in bank]


def ff(sos, zi, x, n=27):
    e = np.concatenate([2 * x[:, :1] - x[:, n:0:-1], x, 2 * x[:, -1:] - x[:, -2:-(n + 2):-1]], -1)
    y = sosfilt(sos, e, zi=zi[:, None] * e[:, :1])[0]
    y = sosfilt(sos, y[:, ::-1], zi=zi[:, None] * y[:, -1:])[0]
    return y[:, ::-1][:, n:-n]


print(f"17-band filtfilt of one window (core1): {best(lambda: [ff(s, z, x) for s, z in zip(bank, zis)]):.2f} ms")
B = np.stack([ff(s, z, x) for s, z in zip(bank, zis)]).astype(np.float32)


def covs(B):
    out = []
    for seg in [B] + np.array_split(B, 2, axis=-1):
        seg = (seg - seg.mean(-1, keepdims=True)).astype(np.float64)
        out.append(seg @ seg.transpose(0, 2, 1) / seg.shape[-1])
    return out


print(f"covariances 17 bands x 3 segments: {best(lambda: covs(B)):.2f} ms; P@C@P on 51: {best(lambda: C1 @ C1 @ C1):.2f} ms")
W = rng.standard_normal((3, 44000)); z = rng.standard_normal(44000); V = rng.standard_normal((44000, 128))
print(f"logistic head 3x44k + two 44k x 128 view projections: {best(lambda: (W @ z, z @ V, z @ V)):.2f} ms")
print(f"deepcopy-size memcpy of 2 x (44k,128) f64: {best(lambda: (V.copy(), V.copy())):.1f} ms")

# stages at prefix t
for t in (190, 380):
    A = rng.standard_normal((t, 51)); Bp = rng.dirichlet(np.ones(3), t); U = rng.standard_normal((t, 128)); U256 = rng.standard_normal((t, 256))

    def em(iters=4):
        m, sd = A.mean(0), A.std(0); Z = (A - m) / sd; Q = Bp
        for _ in range(iters):
            mu = (Q / (Q.sum(0, keepdims=True) + 1)).T @ Z
            Cc = np.cov((Z - Q @ mu).T); Ci = np.linalg.inv(Cc)
            Wm = Ci @ mu.T; L = Z @ Wm - 0.5 * (mu * Wm.T).sum(1); Q = Bp * np.exp(L - L.max(1, keepdims=True)); Q = Q / Q.sum(1, keepdims=True)

    def tsem():
        m, sd = U.mean(0), U.std(0); Z = (U - m) / sd
        sel = [np.arange(c, t, 3)[: max(2, t // 6)] for c in range(3)]
        mu = np.stack([Z[i].mean(0) for i in sel])
        Cc = np.cov(np.concatenate([Z[i] - mu[c] for c, i in enumerate(sel)]).T); np.linalg.inv(Cc)

    def contrast():
        med = np.median(U256, axis=0); mad = np.median(np.abs(U256 - med), axis=0); np.clip((U256 - med) / mad, -5, 5)

    lst = [u for u in U256]
    print(f"t={t}: SessionEM adapt (4 iters, 51 cols) {best(em):.2f} ms; TSEM adapt (128 PCs) {best(tsem):.2f} ms; "
          f"contrast gate medians (256 PCs) {best(contrast):.2f} ms; np.asarray(list of t views) {best(lambda: np.asarray(lst)):.2f} ms")

s = rng.standard_normal(480); ev = butter(4, 8, btype="low", fs=120, output="sos")
print(f"SessionEM.features: 2 x sosfiltfilt on one trace: {best(lambda: (sosfiltfilt(ev, s), sosfiltfilt(ev, s))):.2f} ms")
own = [butter(4, [lo, lo + 6], btype="band", fs=120, output="sos") for lo in (1, 4, 8, 13, 20, 30)]
slow = butter(2, 4, btype="low", fs=120, output="sos")


def observe():
    for so in own:
        b = sosfiltfilt(so, x, axis=-1); b @ b.T
    sosfiltfilt(slow, x, axis=-1)


Co = spd(6)


def members():
    w, Vv = np.linalg.eigh(Co); Wh = np.einsum("bij,bj,bkj->bik", Vv, 1 / np.sqrt(w), Vv); Cw = Wh @ Co @ Wh
    w, Vv = np.linalg.eigh(Cw); np.einsum("bij,bj,bkj->bik", Vv, np.log(w), Vv)


print(f"own-space observe (6 bands assumed + slow, scipy sosfiltfilt): {best(observe):.2f} ms; log_members (2 eigh + 2 three-operand einsum): {best(members):.2f} ms")
from scipy.signal import savgol_filter
tr = rng.standard_normal((4, 480))
print(f"eye traces: 4 x savgol_filter(25, 3): {best(lambda: [savgol_filter(a, 25, 3) for a in tr]):.2f} ms")
Cw = spd(1)[0]
print(f"input-stream whitening (eigh 41 + einsum on 41x480), per stream: {best(lambda: (np.linalg.eigh(Cw), np.einsum('ij,njt->nit', Cw, x[None]))):.2f} ms")
