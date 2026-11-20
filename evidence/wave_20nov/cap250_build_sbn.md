# cap250 SessionBN families: build card (20 Nov 2026)

Code: `cleanslate/cap250/families/{sbn_fbc_ea,sbn_fbc_raw,sbn_sinc,sbn_fbl}.py`, shared `_sbn.py`, parity tool `_sbn_parity.py`.
Recipe: `experiments/session_bn.make(cfg)` fitted on the source rows (load_source order, 40 epochs, unchanged), then one
`to_stream` + device-free trace per latent prior (code of `scripts/export_sbn.py`; no PROTEUS-dependent array is inherited).
Keys `net<I>_{align,k,G,scale,shrink=1e-3,sjit_file}`, I = 35/41 (fbc_ea k2/k5), 47, 53, 59 plus seed. Each export fails
loudly if the trace is not stateful with zero state, or differs from the eager SessionBN by >=1e-4 on two sessions. The CUDA
requirement and the m7r init path (resolved under CAP250_ASSETS) fail loudly. `CAP250_SMOKE=1` = 2 epochs.

All runs: Leonardo boost_usr_prod (A100-64GB), account AIFAC_P02_1675, via `worker.py` (receipts), code = working tree,
uncommitted at run time (receipt revision strings "uncommitted-working-tree"/"full"); release dir
`/leonardo_work/AIFAC_P02_1675/ni26_cap250_sbn_20nov`. Galvani smoke 3141353 was cancelled (QOSMaxJobsPerUser).

## Jobs
- Reduced smoke, nine people P02-P10, seed 0, four families: 64930881 (86 s), 4/4 PASS, receipts + diagnostics written.
- All-ten, seeds 0-5 per family: 64931053 (fbc_ea) 64931057 (raw) 64931060 (sinc) 64931063 (fbl).
- Nine-person fold P02-P10, full recipe, seeds 0-5: 64931054, 64931058, 64931061, 64931064; all PASS (receipts in `out/fold9`).
- Parity vs archive `runs/sealed/cand_v51b_own`: 64931081, 64931083, 64931086, 64931089. Fit-vs-checkpoint diagnostics 64931991.

## Parity (all-ten fresh fit vs archived members, first 4 sessions = 755 windows, streamed with state reset)
| Family | G, scale, align, k, shrink, sjit_file | trace probabilities vs archive |
| --- | --- | --- |
| sbn_fbc_ea (k2, k5, 6 seeds) | identical (G rel diff 0, scale 0) | max abs diff < 5e-4 (prints 0.000), 0 flips |
| sbn_fbl (6 seeds) | identical | max abs diff < 5e-4, 0 flips |
| sbn_sinc (6 seeds) | identical | max 2.0e-3 to 6e-3, 1 argmax flip in 6x755 (s3); GPU nondeterminism |
| sbn_fbc_raw (6 seeds) | identical | **max 0.058-0.094, 17-24 flips per 755 (2-3 %)** |

The raw mismatch is not a nondeterminism effect. Diagnostic 64931991: two fresh all-ten fits of `fbcnet_raw_sbn` s0 are
bitwise equal to each other and to the stored `runs/sbn/fbcnet_raw_sbn-s0.pt` (Leonardo, slurm_job 64286447, A100), with the
same G/scale. Diagnostic (diag2): the archived `sbn_fbcnet_raw_k20_s{0,1,2}.pt` differ from that checkpoint in exactly 8 of 26
float parameters, all in `spatial_conv` (bias diff up to 0.11, conv weight up to 0.15, BN up to 0.07); all others agree.
So the archived raw traces come from a different training state than the checkpoint cache on Leonardo (open: Galvani cache
diag job 3141490 pending in cpu-galvani; provenance of the 13 Nov "twin" retrain is the likely cause, review 13 Nov). The family
is bitwise faithful to the cached checkpoint but NOT shown to reproduce the archived raw trace; no claim of raw parity.

## GPU time (A100, wall of the whole six-seed job, one GPU, includes ~15 s imports per seed)
| Family | all-ten | nine-person | GPU-h per fold (6 seeds) | SPEC range |
| --- | ---: | ---: | ---: | --- |
| sbn_fbc_ea | 3 m 33 s | 5 m 11 s | .06-.09 | .05-.10 |
| sbn_fbc_raw | 4 m 33 s | 4 m 30 s | .075 | .05-.10 |
| sbn_sinc | 6 m 43 s | 6 m 31 s | .11 | .10-.20 |
| sbn_fbl | 3 m 30 s | 4 m 27 s | .06-.075 | .05-.10 |
Total about 0.31-0.35 GPU-h per fold for the four families.

## Contract notes for the lead
- Package alignment `k` equals the net's EA prior (5.0), as in the archive; the file-name k is the latent prior only.
- `components.expected_files` and `assets` for SBN match what the families write; `diagnostics/` is skipped by worker.
- `source.family_args` limits seeds to range(6) for all four, matching FAMILIES.
