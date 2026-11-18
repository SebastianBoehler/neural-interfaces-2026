# Sinc provenance for the held-new full0 replay

The original Sinc seed-0 producer is Galvani 3103245 task 2. Slurm records an A100 on cn218, eight CPUs and 64 GiB on 14 November 21:19:41–21:26:45 CET. Its submission line uses eight OpenMP, OpenBLAS and MKL threads, the workspace Python, `KIND=v51b`, `ARMS=s15,r15`, `CAPS=100,150,250`, `SCHEME=full` and `PSEED=0`. The archived stdout identifies the Sinc family, seed 0, all-LOSO P01–P10 and the v5.1b pseudo-label file prefix `8fb5d9296084`. Original GPU SKU, device ordinal, and effective cuDNN/TF32 flags are unavailable.

| Pinned evidence | SHA-256 |
| --- | --- |
| `runs/diag/arr/ftrcfull0-3103245_2.out` | `8d3ee93252a34fa9a01c5403d9c9f991ba755b9826b19fd9129bcf00cb5e173c` |
| §257 full0 Sinc seed-0 frozen bank | `8adafb1efff1cb0f23ccd640a1e2dc2f0b5f9682fd79a0ddf579a9f2592102ba` |
| §257 full0 Sinc seed-0 r15c250 bank | `9e4a042800e395c838b47490b13b3317719667763ef20ba7af241bccfa80f438` |
| `runs/pretrain/sincshallow_ea-m7r-s0.pt` | `f28f2865cf666bc85d872aacea2f2fb147dfc3d3835d6bf3ff2df55aa949ab8f` |

The canonical Galvani §257 output files match the later §281/§430 archive copies byte for byte. The pretraining trunk hash matches the [independent 16 November handoff](../../../neural-interfaces-2026/evidence/s281_16nov/sinc_reconstruction_handoff.md). The six held-new training source files `deep.py`, `features.py`, `run_cv.py`, `session_bn.py`, `protocol.py`, and `dg.py`, plus `ttt5.py`, `ttt7.py` and `ftrel.py`, match repository snapshot `12944eb3`. The live `ttt18.py` matches `17697223`. The only difference through `12944eb3` adds the unused cum10 configuration. The original uncommitted runtime file hashes were not recorded when 3103245 started. The existing workspace environment was prepared before the original run and still carries PyTorch 2.14.1+cu126, Braindecode 1.8.1, NumPy 2.5.4 and cuDNN 9.10.2.21. The original stdout has the same dependency paths and warnings.

The original family selector is `sincshallow_ea_ptm7r_rot05_sbn` with six fixed pretraining seeds. The actor uses fit-only 40-epoch SessionNet training with k=5 input EA, SessionBN grouping 32, latent k=20, and rotation 0.05. The original harness also ran a recording-order diagnostic forward before the shuffled forward. The retained implementation uses evaluation-mode running buffers and a local prefix sum for that forward. It does not update the buffers. The old refit target file can affect adapted outputs and cannot explain a frozen-input discrepancy.

Thirteen distinct archived full0 seed-0 Sinc frozen arrays are keyed to the same 2,655 canonical rows as the LPC bank. All matched the original LPC P01 decisions on 190 rows, while every maximum absolute posterior difference exceeded 2e−5. The P01 error range was 0.00047594–0.00451457. Across all rows, each comparison changed 2–8 decisions. Two original Slurm receipts bind these historical differences to real seed-0 runs:

- `3096655_2`, RTX 2080 Ti cn102, eight CPU threads. The v5.1b-label full0L Sinc array differs by 0.00047594 max on P01, with unchanged decisions.
- `3098384_2`, RTX 2080 Ti cn111, eight CPU threads. Its timestamp matches the `runs/diag/ttt/probs18` full0R copy, whose P01 max is 0.00451457 with unchanged decisions. The separate earlier `runs/ttt/probs18` full0R copy also differs and must keep its own provenance.

The fresh 4-thread and 8-thread A100 errors of 0.000615 and 0.002609 lie within that historical range. The same-node eight-thread retry, 3131432 task 2, finished on cn218 at 21:22:39 CET on 18 November. It preserved all 190 P01 decisions and failed the same 2e−5 gate with maximum frozen error 0.00266153046 before reaching the refit. Its stdout SHA-256 is `905efc1b4d92d443efd2dedb0fd7030b6c352d4d1721b0621bf8c579f524a6f3`. The evidence supports fit/runtime variability, with its exact cause unresolved. The archived code saves posteriors and row indices. It does not save the fitted state. The inspected canonical run directories had no Sinc checkpoint tied to the original held-new P01 fit. Returning visit-excluded and deployment Sinc weights exist with different source folds and cannot authenticate this fit.

## Implemented matched seed-0 bridge

The registered repair replaces only the active Sinc seed 0 across the ten original all-LOSO folds. It preserves authentic archived outputs for the other five frozen seeds. The source actor persists each fitted state and same-fit recording and full0 predictions. The bridge authenticates each fold and the complete bank. It keeps the original recording-order recognition calibration fixed as a shared reference.

The live Sinc mix changes by one-sixth of the fresh-minus-archived full0 seed-0 probability difference before the existing normalization. Both arms share that same replacement. Both incumbent shadows, recognition traces and strict-past refit targets are rebuilt. The archived exact zero remains a separate instrument. Any recognized session blocks the paired refits until its routed tail is reproduced.

The four active refits must consume these new arm-specific targets. Sinc uses its fresh seed-0 state. The other three families use authenticated restored source fits. Their adapted-minus-same-fit-frozen deltas enter the paired system through the original seed-0 policy. Earlier LPC and regenerated targets belong to a different upstream input. The held-new full0 result remains a fresh matched instrument.
