# Arm c final decision · 21 November 2026

**Use the SessionEM-only arm c archive.** The 21:15 CET vote selected it subject to the exact-archive checks. The fresh PROTEUS and Scherer replays supported that decision. The team left Codabench untouched. [Board](../../../ni26-cleanroom/BOARD.md) · [Preregistration](../../docs/LEDGER.md#arm-c-fresh-proteus-confirmation--21-november)

`runs/sealed/final_zips/final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing_recognised_em_half_diet_c.zip`

SHA-256 `cba9b208fbff7ce416efd8b5ce68ae1ad0771980d84e754433bd33e9996d0bca`. Size 417,253,955 bytes. The 43 ZIP members match em_half in name and order. Content hashes match for every member except `submission.py`. Arm c lowers the recognized SessionEM emission weight from 0.5 to 0.25 after the native fit. The replay and archive both apply normalized `p^0.75 * q^0.25` when SessionEM is active. The native CountFilter and TSEM still run. [Exact archive qualification](diet_decomposition.md#promotion)

## Fresh PROTEUS orders 72120–72123

Leonardo smoke `65059840` passed v3 and reproduced live arm c labels on the first 84 P10 visit-3 windows in both schemes. Array `65060147` completed all 48 tasks with source `f9d040f1`. Collector `65060168` authenticated all 96 shards and delivery rows. The report is `cap250/runs/gpt-arm-c-confirm-21nov/report.json`, SHA-256 `13fc516ef29aee94c13e67bf059baca968d492c95f9013d9945ecb262b202f85`.

Differences are balanced-accuracy percentage points against em_half. Intervals use 20,000 paired people-then-orders draws.

| Delivery | Returning Δ [80% interval] | 50/50 Δ [80% interval] | P04 / P10 Δ | Returning labels changed | Held-new labels changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-session | +0.179 [−0.002, +0.358] | +0.089 [−0.001, +0.179] | +0.197 / +0.160 | 48 / 3,048 | 0 / 10,620 |
| Within-run | +0.160 [−0.094, +0.395] | +0.080 [−0.047, +0.197] | +0.260 / +0.060 | 41 / 3,048 | 0 / 10,620 |

All 32 returning streams recognized correctly. None of the held-new streams recognized. The pooled returning delta was +0.169 pp, above the board's −0.15 pp cutoff. The repeated positive points are small and both 80% intervals include zero. This replication holds out delivery orders and reuses the same participants. [Original PROTEUS screen](diet_decomposition.md#proteus-held-7180871815)

Scherer orders 71872–71879 replicated the transfer check on the same nine outside participants. Returning gained +0.089 pp [−0.109, +0.293] full-session and +0.114 pp [−0.058, +0.289] within-run. [Fresh Scherer receipt](diet_decomposition.md#scherer-ni-v-b-ret-orders-7187271879)

## Production qualification

- Galvani `3150612` on RTX 2080 Ti cn122 changed 0/379 held-new labels versus em_half. Max posterior delta was 5.14e−6. The same-node em_half A/A control `3150635` observed 7.70e−6 and 0/379 changed labels. A/A receipt SHA-256 `4fede8836566c841d82073bb6577bde21ee77dad56d1799558cb45eb85ac5c98`.
- Galvani `3150613` passed official warm-up with scores of 1.0000 in both replays of each archive and all package files unchanged. `3150614` measured arm c at 167.337 ms/window versus 166.761 for em_half on the same 1,131 windows (+0.346%), giving a cap250-anchored estimate of 3,139.26 s. All four completed follow-up placements passed the timing condition. The largest observed slowdown was +0.515% and the largest arm c projection was 3,171.8 s, 428.2 s below the 3,600 s budget. [Timing and warm-up receipts](diet_decomposition.md#promotion) · [Host repeats](diet_decomposition.md#galvani-final-fill)
- Leonardo `65057672` ran signed em_half twice for 20,083 calls over 106 fresh sessions per arm. Both completed with no Solver, stage or Team B graph errors. Compute time was 2,835.68 and 2,852.28 s across separate A100 nodes. Peak CUDA reserved was 2.296 GB. The last tenth changed by +2.26% and −1.56% versus the second. This test checks sustained execution of the parent. Arm c's runtime outside the tested RTX 2080 Ti workload and its sealed score remain unmeasured.
