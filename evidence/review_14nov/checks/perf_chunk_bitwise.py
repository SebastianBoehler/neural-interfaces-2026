"""Is a chunked (and threaded) batched eigh + rebuild bitwise equal to the unchunked one? Which step differs?"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
torch.set_num_threads(1)
try:
    from threadpoolctl import threadpool_limits; _l = threadpool_limits(limits=1)
except ImportError:
    pass
rng = np.random.default_rng(0)
A = rng.standard_normal((51, 41, 120)); C = A @ A.transpose(0, 2, 1) / 120 + 1e-3 * np.eye(41)
eig = lambda C: tuple(t.numpy() for t in torch.linalg.eigh(torch.from_numpy(C)))
reb = lambda w, V: (V * np.log(w)[:, None, :]) @ V.transpose(0, 2, 1)
w0, V0 = eig(C); L0 = reb(w0, V0)
for k in (2, 3, 4, 6, 51):
    parts = np.array_split(np.arange(51), k)
    wv = [eig(C[i]) for i in parts]
    e_eq = all(np.array_equal(w, w0[i]) and np.array_equal(V, V0[i]) for (w, V), i in zip(wv, parts))
    r_eq = np.array_equal(np.concatenate([reb(w0[i], V0[i]) for i in parts]), L0)
    pool = ThreadPoolExecutor(k if k < 8 else 8)
    thr = [all(np.array_equal(np.concatenate(list(pool.map(lambda i: reb(*eig(C[i])), parts))), np.concatenate([reb(*eig(C[i])) for i in parts])) for _ in range(5))]
    pool.shutdown()
    print(f"chunks {k}: eigh chunked == unchunked {e_eq}; rebuild matmul chunked == unchunked {r_eq}; threaded == serial at the same chunking {thr[0]}")
print("per-matrix rebuild with a plain loop == stacked matmul:", np.array_equal(np.stack([(V0[b] * np.log(w0[b])) @ V0[b].T for b in range(51)]), L0))
