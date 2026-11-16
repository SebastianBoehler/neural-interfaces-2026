# §307 source-person train-for-adaptation pilot

Registration: `aaaca45`, `docs/LEDGER.md` §307. The strong reconstructed core has
44,239 features; source-episodic training uses only eight-person population
heads and source-person query labels. The outer held person is excluded from
both the source fit and meta-training. Target-label scores have not been read.

Synthetic tests passed for a three-step differentiable update, active
meta-gradient, norm bound, frozen slow coefficients, zero identity and session
reset. The real-data pilot **3115164** is queued on Galvani in
`/mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/smoke`. It fits the
P01/P02 eight-person anchor using the frozen §305 CPU solver, generates one
causal source stream, and performs scalar and block meta-gradient steps. Its
source manifest SHA is `9085404ee45acf076150243e6bee008cbbd24b7d56836031b3787c0de63c47eb`;
the pinned cache manifest SHA is
`305144ee6d448212d7559f5516ed52f47bbccaf8b52b9a200ecfa3a4a971094d`.
Pilot elapsed time is not yet measured.

Release **3115173** depends on a successful pilot. It independently verifies
its evidence and queue capacity before scheduling 90 cross-fits at %2, a complete
source/provenance verifier, and ten source-only meta-fit tasks at %2. It records
actual job IDs in the production workspace's `release.intent` and retries when
reserved queue capacity is unavailable. The production dispatch manifest SHA is
`921397d815bb225c7ce54cf04a1dadfe35ef1a53e80de9829a0805cc57b751b0`.
A failure blocks downstream work. The target-stream generation and full-chain
six-CP evaluation remain to implement and validate; they are not queued.

## Explicit CPU route

The initial CPU pilot 3115188 failed before fitting: the dataset stores full
`Dreyer2026Proteus/P02` subject IDs, while the first guard expected `P02`.
A regression now normalizes these IDs to the protocol fold names. The old GPU
pilot remains independent; its old-source release was canceled before it could
submit any child.

Corrected CPU cross-fit pilot 3115208 passed 0:0. Fit and one complete within0
causal source stream took 44.6 s after data loading. For three same-session
source windows, versus original NumPy/SciPy equations, relative feature L2
error was 2.21e-14, maximum absolute feature error 2.45e-13, and probability
error 2.47e-14. No non-tied predicted labels differed; prefix and reset passed.
CPU source meta pilot 3115210 passed 0:0; its scalar and block second-order
meta-gradients were nonzero, and a single episode took 0.096 and 0.063 s after
setup. These are pilot timings, not a full-wave estimate.

Immutable CPU production is
`/mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production_cpu_v2`.
Its source, CPU route, meta route and dispatch manifest SHAs are respectively
`b43dc93690f760688973940df6a999c71b8b4f899c9fedc283c93f094da779dc`,
`8be4a86466cfc9e53e83455b66cdad80ced4be8bbd9f1223364f8f92f16f0172`,
`124d2bc7ab6d78742b23ee5eae5ac4766197d4741b288bf27d42d49bda1bb0f8`, and
`2e5f948a39bc022558904ff9485800ecc16171898e5732a31ae71f40eab747e8`.
CPU cross-fit array 3115215 runs %4; complete hash/route verifier 3115216
follows it; CPU meta array 3115217 runs %2 after verification and CPU meta
gate 3115213. All execute source-only work; no target result has been scored.
The separately queued GPU v2 pilot 3115209 is an independent check, not a gate
or automatic fallback. Target-stream and full-chain integration are still open.
