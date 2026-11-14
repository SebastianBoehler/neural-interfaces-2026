"""Order-of-magnitude cost of the CPU parts that step62 runs twice (synthetic data; only array headers and scalars of the
package are read, no net is loaded). Mac CPU, one thread; the machine is busy, so read the numbers as rough."""
import io, sys, time, types, zipfile, importlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]
pkg = types.ModuleType("src62h"); pkg.__path__ = [str(ROOT / "lean/src62h")]; sys.modules["src62h"] = pkg
personal, stages, extras, inputs = (importlib.import_module(f"src62h.{m}") for m in ("personal", "stages", "extras", "inputs"))

zf = zipfile.ZipFile(ROOT / "runs/sealed/cand_v62_r2hL/weights.npz")
infos = {i.filename[:-4]: i for i in zf.infolist()}
rng = np.random.default_rng(0)


def header(k):
    with zf.open(infos[k]) as f:
        v = np.lib.format.read_magic(f)
        return (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f)[0]


def val(k):
    return np.load(io.BytesIO(zf.read(infos[k])), allow_pickle=False)


def timeit(fn, n=30):
    fn(); t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t))


X = (20 * rng.normal(size=(64, 41, 480))).astype(np.float32)

# --- own-space stage: observe (every window) and log_members (routed windows) ---
z = {k: (val(k) if infos[k].file_size < 4096 else rng.normal(size=header(k))) for k in infos if k.startswith("own_")}
A = rng.normal(size=(10, 9, 41, 41)); z["own_prior"] = A @ A.transpose(0, 1, 3, 2) / 41 + np.eye(41)
z["own_t_sd"] = np.abs(z["own_t_sd"]) + 1.0
own = personal.OwnStage(z)
i = [0]
def observe():
    own.observe(X[i[0] % 64]); i[0] += 1
print(f"OwnStage.observe            {timeit(observe):6.2f} ms  (bands {len(own.sos)})")
print(f"OwnStage.log_members        {timeit(lambda: own.log_members(3)):6.2f} ms")

# --- one tail (rebalancing -> SessionEM -> TSEM) at t = 100 and 300, 48 and 51 view columns ---
zs = {k: val(k) for k in infos if k.split("_")[0] in ("em", "rb", "eb", "tsem") and infos[k].file_size < 4096}
for cols, name in ((0, "Tail (48 columns)"), (3, "Tail + 3 eye columns")):
    for T in (100, 300):
        tail = stages.Tail(zs)
        if cols:
            f0 = tail.em.features
            tail.em.features = lambda x, f0=f0: np.concatenate([f0(x), rng.normal(size=3)])
        for t in range(T):
            p = rng.dirichlet(np.ones(3) * 2); tail.step(X[t % 64], rng.normal(size=128), p)
        def one():
            n = len(tail.em.A); tail.step(X[n % 64], rng.normal(size=128), rng.dirichlet(np.ones(3) * 2))
        print(f"{name:27s} {timeit(one, 20):6.2f} ms  at t = {T}")

# --- eye traces and the rest member's input stream (their nets are not timed here) ---
e = extras.EyeNet.__new__(extras.EyeNet)
ix = {c: k for k, c in enumerate(extras.CH_NAMES)}
e.idx = {k: [ix[c] for c in cs] for k, cs in extras.EYE_SETS.items()}; e.n, e.s1, e.s2 = 0, 0.0, 0.0
print(f"EyeNet.traces               {timeit(lambda: e.traces(X[3])):6.2f} ms")
G = np.cov(rng.normal(size=(41, 2000)))
s = inputs.InputStream(dict(align=True, k=5.0, G=G, scale=1.0, shrink=1e-3))
print(f"InputStream (aligned)       {timeit(lambda: s(X[5])):6.2f} ms  (einsum covariance + one eigh)")
xc = X[5] - X[5].mean(-1, keepdims=True)
print(f"  of which einsum covariance {timeit(lambda: np.einsum('nct,ndt->ncd', xc[None], xc[None], dtype=np.float64)):6.2f} ms")
