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
