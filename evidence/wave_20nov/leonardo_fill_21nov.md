# Leonardo Team B · 21 November 2026

[Preregistration](../../docs/LEDGER.md#leonardo-fill--21-november-registered-before-71400-scoring) · [Dispatch](DISPATCH_21NOV.md)

The organisers confirmed that sealed evaluation averages returning and held-new scores equally. This report uses 50/50. Model loading is outside the 60-minute compute budget, and a timeout ends as Failed. The user’s shipping target is 3,450 s on the worst normal proxy node. [Current rules](../../docs/STATE.md)

## Finalists read before the new orders

The existing finalists head-to-head used 71200–71215. On the real folds, nine25 gained **+0.67 pp [+0.34, +0.97] held-new** and **+0.98 [+0.72, +1.25] returning** over hard. Nine15 gained +0.44 [+0.20, +0.65] and +0.76 [+0.51, +1.05]. Nine25 minus nine15 was +0.23 [+0.05, +0.41] and +0.22 [−0.03, +0.44]. `ba_terminal` minus nine25 was +0.00 [−0.20, +0.20] and +0.61 [+0.42, +0.80]. The existing `ba_terminal` ZIP exceeded the 3,600 s RTX 2080 Ti proxy on both nodes and triggered the Scherer returning flag. [ZIP qualification](final_ba_terminal.md)

S1 resolves the conditional seed lane. On the paired 71124–71131 replays, the nine25 gains were +0.97/+0.59 pp (seed 0, 50/50 +0.78), +0.95/+0.60 (seed 1, +0.78), and +0.93/+0.80 (seed 2, +0.87), held-new/returning. The three-seed mean was **+0.95 [+0.25, +1.53] / +0.67 [+0.36, +0.96]**, 50/50 +0.81 pp. All three met the S1 robustness rule. Every paired seed difference included zero in both cohorts.

Leonardo collectors 65011278 and 65011279 completed. Their receipts are `cap250/runs/leonardo-fill-21nov/finalists-seeds.json` and `finalists-h2h.json`. Initial collector jobs 65011239 and 65011241 stopped before reading scores because the finalists release lacked the collector source. The retry staged the collector under this run and used the original finalists tapes.

## Fresh dose and returning comparison

Orders 71400–71431 are disjoint from the finalists replays. The instrument uses the same ten held-new people and P04/P10 returning people. Every interval is paired and resamples people and orders with official cell weights.

Source `4a3d7872`. Smokes 65011374–65011376 passed v3 replay and matched all 1,695 pinned labels across the three arms on P01 visit 1, P04 visit 2 and P10 visit 2. The zero mixer was exact in the same call, both doses reached the live mixer, and the scorer watched for degraded terminal stages. At 08:08 CET the two 96-task arrays 65011605/65011606 were running or queued. Collectors 65011607, 65011608 and 65011609 wait for the first, second and both arrays respectively. The collector reproduced the historical Lane TB-verify gains for both cohorts and both doses from the original 71124–71131 predictions.

Both 96-task arrays and all three reports completed by 08:53 CET. The pooled `dose.json` SHA-256 is `c41c3f47f7eed1744605f298dc969df4c203f893a7345abd9633012e93aaf60b`.

| 32-order contrast | Held-new pp [80%] | Returning pp [80%] | P04 / P10 pp | 50/50 pp [80%] |
|---|---:|---:|---:|---:|
| nine25 − hard | +0.87 [+0.45, +1.24] | +0.82 [+0.58, +1.04] | +0.66 / +0.98 | +0.85 [+0.61, +1.06] |
| nine15 − hard | +0.66 [+0.42, +0.87] | +0.65 [+0.48, +0.81] | +0.58 / +0.72 | +0.65 [+0.51, +0.78] |
| nine25 − nine15 | +0.21 [−0.01, +0.41] | +0.17 [+0.02, +0.30] | +0.08 / +0.26 | +0.19 [+0.06, +0.31] |

Absolute BA by held-new / returning: hard 73.32% / 79.31%, nine15 73.98% / 79.96%, nine25 74.19% / 80.13%. Nine25 returning BA is 70.67% on P04 and 89.59% on P10. The two 16-order nine25 returning gains were +0.55 [+0.22, +0.85] and +1.09 [+0.84, +1.33]. Their 0.542 pp gap exceeds the registered 0.5 pp stability rule. The returning dose advantage passes the registered rule because its pooled lower bound is positive and both blocks favored nine25 (+0.06 and +0.28). Held-new dose advantage is undecided on these orders.

At 09:23 CET the queue had five running and one pending. I registered a second disjoint 32-order block (71600–71631) to check the returning order dependence and restore the queue. It uses the same folds, bank and inference path. Source `674930c0`, run `cap250/runs/leonardo-fill-retest-21nov`. The collector will report the new 32 and a pooled 64 with the original orders.

Retest smokes 65014371–65014373 passed all original pinned labels, zero and v3 checks. Leonardo accepted arrays 65014554/65014555 (96 tasks each) and collector 65014556. The 09:32 CET submission snapshot showed 25 running and 225 pending across the account, plus one completing.

| Retest contrast | Held-new pp [80%] | Returning pp [80%] | 50/50 pp [80%] |
|---|---:|---:|---:|
| 71600–71631 · nine25 − hard | +0.83 [+0.36, +1.20] | +1.02 [+0.81, +1.23] | +0.92 [+0.67, +1.13] |
| 71600–71631 · nine15 − hard | +0.69 [+0.38, +0.94] | +0.76 [+0.59, +0.93] | +0.73 [+0.55, +0.88] |
| 71600–71631 · nine25 − nine15 | +0.14 [−0.07, +0.32] | +0.26 [+0.12, +0.40] | +0.20 [+0.07, +0.32] |
| Pooled 64 · nine25 − hard | +0.85 [+0.42, +1.20] | +0.92 [+0.76, +1.07] | +0.88 [+0.66, +1.07] |
| Pooled 64 · nine15 − hard | +0.68 [+0.42, +0.88] | +0.71 [+0.58, +0.82] | +0.69 [+0.55, +0.81] |
| Pooled 64 · nine25 − nine15 | +0.17 [−0.02, +0.34] | +0.21 [+0.11, +0.30] | +0.19 [+0.08, +0.29] |

The two retest halves gave nine25 returning gains of +1.13 and +0.91 pp. Their 0.22 pp gap meets the stability rule on the retest. In the pooled 64, nine25 reached 74.34% held-new and 80.12% returning BA, including 70.70% on P04 and 89.53% on P10. The pooled direct returning dose advantage is +0.18 pp on P04 and +0.24 pp on P10. Collector 65014556 completed at 10:18 CET. `doseall.json` SHA-256 `30d3289ace2858f94dda237e72d52860b5c6ddec38ecc5280d15ad95d123cb6e`.

## Bank growth within +8 ms/window on A100

Source `be01260a`. The three arms are `swap25`, `fbc12` and `ba_six`. The first changes three Team B members at fixed count. The second adds three fitted FBC members. The third uses six Team B members and the existing graph-based MANAS A1 terminal path. All four variants assembled from the real P01 banks locally, and the signed lead source reproduced byte-for-byte from its production bank. Growth live smokes 65011663–65011665 passed at 08:15 CET. They matched the lead and zero-control labels on 565 windows each, verified both CUDA graph sizes, and checked the exact MANAS terminal blend and session-local state.

At 08:15 CET controller `e8db6272` released nine remaining fold fixtures (65011888), six timings (65011889) and 48 screen tasks on 71500–71507 (65011890). Freeze 65011891 depends on the timings and the screen. Release 65011892 schedules held-out work on 71508–71523 only if a variant qualifies, then schedules its report. Slurm writes the IDs to `grow/screen-jobs.tsv` and `grow/held-jobs.tsv`.

| Screen vs nine25 | Held-new pp [80%] | Returning pp [80%] | P04 / P10 pp | 50/50 pp [80%] | Added A100 ms/window |
|---|---:|---:|---:|---:|---:|
| swap25 | +0.00 [−0.19, +0.17] | −0.27 [−0.53, −0.03] | −0.40 / −0.13 | −0.13 [−0.31, +0.02] | −0.89 |
| fbc12 | −0.32 [−0.56, −0.09] | −0.34 [−0.52, −0.13] | −0.29 / −0.39 | −0.33 [−0.50, −0.16] | +0.12 |
| ba_six | +0.03 [−0.25, +0.36] | +0.85 [+0.29, +1.52] | +1.19 / +0.51 | +0.44 [+0.09, +0.84] | +2.23 |

Timing used 757 windows per arm across the four paired A100-SXM-64GB sessions. The `ba_six` pair deltas were −0.65, +2.56, +2.77 and +4.27 ms/window. The reference loaded first at both placements and its load time varied, so the negative load deltas are startup-order diagnostics. The decision uses the counterbalanced full-window times, including fresh-session setup. `selection.json` SHA-256 `f1c8cae836f681d9940c5e7eff03afb11f91efc3693556ab652172d7c0503ff0`.

The 96 held-out tasks (65013453) and report 65013454 completed by 09:26 CET. `ba_six` versus nine25 was **−0.10 [−0.39, +0.16] held-new and +0.70 [+0.41, +0.99] returning** across 16 disjoint orders. P04 +0.71, P10 +0.69, and 50/50 +0.30 [+0.09, +0.50]. This qualifies under the user’s revised rule at 12:15 CET. That rule accepts a positive target-cohort lower bound and a loss of at most 0.30 pp in the other cohort.

## Second bank growth wave

Source `30074abb`, run `cap250/runs/leonardo-fill-roster-21nov`. The screen uses 71700–71715. `triseed25` spreads nine Team B members evenly across three existing fit seeds. `ba_seven` and `ba_eight` retain more of the original Team B bank and keep the same MANAS terminal term. All three assembled from the original trained P01 banks, with fresh signed CPU references. The same +8 ms/window A100 cap and growth retention rules apply. The confirmation orders are 71716–71731.

Smokes 65014561, 65014571 and 65014573 use P01, P04 and P10. Controller 65014651 waits for all three, then checks Slurm completion and the exact replay receipts before submitting the prep, timing and screen jobs. The 09:34 CET account snapshot had 33 running, 193 pending and ten configuring.

| Roster contrast vs nine25 | Held-new pp [80%] | Returning pp [80%] | 50/50 pp [80%] | Added A100 ms/window |
|---|---:|---:|---:|---:|
| Screen · triseed25 | −0.01 [−0.22, +0.17] | +0.12 [−0.12, +0.33] | +0.05 [−0.10, +0.19] | −1.13 |
| Screen · ba_seven | +0.03 [−0.21, +0.30] | +0.60 [+0.30, +0.93] | +0.32 [+0.12, +0.53] | +2.28 |
| Screen · ba_eight | +0.06 [−0.21, +0.34] | +0.51 [+0.21, +0.81] | +0.28 [+0.07, +0.50] | +3.87 |
| Held · ba_seven | −0.19 [−0.38, +0.03] | +0.54 [+0.24, +0.84] | +0.18 [−0.002, +0.36] | +2.28 |

Screen 65015076 and freeze 65015077 completed by 10:39 CET. Held array 65017133 and report 65017135 completed by 11:20 CET. Returning P04/P10 for the held contrast were +0.33 / +0.75 pp. The revised rule also admits `ba_seven` on fold quality. The user prioritized `ba_six`, which has the larger held 50/50 point and lower A100 cost here. The two held contrasts use different orders. `selection.json` SHA-256 `6731aed2fb1d2831c9889154c62f103b63c4a6c4f115a00daf268348abf9e422`. `confirmation.json` SHA-256 `ddf4f5793d394673080ee04b93e8888d89a3974b2085caf8eed22d22fd898506`.

## Fused ba_six package and afternoon qualification

Both own-job queues were empty on the 12:11–12:15 CET check. Galvani run `ni26_ba_six_fc_20261121` and Leonardo run `cap250/runs/leonardo-fill-ba-six-21nov` resumed with the new priorities. The signed parent is fused nine25 SHA-256 `7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07`. The builder keeps the first two members of each Team B family and copies the signed MANAS source, self-tests and weights. Four of the six Team B members use fused convolution on CUDA.

| New archive | SHA-256 | Use |
|---|---|---|
| `final_stack_hard_safe_st_team_b_ba_six_fc.zip` | `fc2acf89bc194e7e576dd9fa9b6a6858207d06fe98396a1e4e1f785c04e9a5aa` | Production candidate, 417,253,492 bytes |
| `ba_six_fc_niv.zip` | `3c4796d5143d307fdfad0004557b3ceffbd67f2ff18d76068b4e4083d8eaaf29` | Scherer-free validation |

Galvani build 3147531 completed at 12:21 CET with source `7b845dc0`. Smoke 3147532 gates 379-window CPU zero 3147533, active fused/unfused identity 3147534, and official warm-up pair 3147535. E2E jobs 3147536–3147541 request cn111, cn114, cn116, cn117, cn122 and cn123 respectively. NI-V smokes 3147542/3147543 gate controller 3147545. The Scherer prior for the original Team A terminal is **B-ret −0.52 [−1.01, −0.02] pp**, an external flag under the current decision rule. [Original Scherer result](final_ba_terminal.md)

The exact Galvani production ZIP is mirrored in local `runs/sealed/final_zips/` with its SHA verified. The Mac build had different `team_b.pt` serialization bytes and a maximum absolute CPU reference difference of 6.0e−8. All six trained state dictionaries matched bitwise. The Galvani bytes are the candidate for every qualification.

Production smoke passed on 144 windows with zero fused/unfused label changes, both graphs active and the exact MANAS terminal formula. Active identity 3147534 completed on 379 windows. NI-V smokes passed, then controller 3147545 released 36 paired B-new tasks 3147546, nine paired B-ret tasks 3147547 and report 3147548.

Official pair 3147535 completed four replays with stable predictive scores, unchanged package files and no v6 errors. Cap250 was 99.4579% and the candidate 100.0000%. The runner exited 1 because the 0.5421 pp increase exceeds its symmetric 0.5 pp tolerance. Registered a second official pair against the signed fused lead, whose prior official BA is 99.5923%. Job 3147630 uses the same tolerance and official scorer. [Fused lead’s official pair](teamb_overhead.md)

Leonardo source `e00c77b5`. All three fused fold smokes 65023503–65023505 passed v3, exact pinned rows and labels, zero and active graph checks. Controller 65023506 released nine prep tasks 65023987, 96 replays 65023988 on 71800–71815 and paired collector 65023989.

### Exact archive validation

CPU zero 3147533 passed at 12:42 CET. Disabling Team A matched the signed-lead-derived six-member control bitwise on 379 windows. Disabling both Team A and Team B matched signed hard bitwise on the same 379 windows. Both checks cover labels, captured scores and stage activity. GPU identity 3147534 changed 0/379 labels when fused convolution was disabled. Maximum score difference was 6.0e−8.

The official fused-lead pair 3147630 passed at 12:54 CET. Both repeats gave 99.5923% lead BA and 100.0000% candidate BA (+0.4077 pp). Each replay processed four v6 streams, kept package files unchanged and logged zero v6 errors. Official warm-up measures ingestion and repeatability on the public rehearsal.

### Scherer NI-V

All 36 B-new and nine B-ret matched tasks completed. Collector 3147548 used eight orders per scheme (71840–71847) and the same visit-1 enrollment on both returning arms. The candidate’s MANAS weights exclude Scherer. Values are paired percentage-point deltas for fused `ba_six` minus fused nine25.

| Delivery | B-new pp [80%] | B-ret pp [80%] | 50/50 pp |
|---|---:|---:|---:|
| Full-session | +0.06 [−0.43, +0.54] | −0.40 [−1.07, +0.23] | −0.16 [−0.57, +0.24] |
| Within | −0.03 [−0.36, +0.30] | −0.19 [−0.74, +0.36] | −0.11 |

Primary absolute BA is 47.97% versus 47.91% held-new, and 55.39% versus 55.79% returning. The worst B-ret person lost 3.85 pp. The original full Team A terminal returned −0.52 [−1.01, −0.02] pp. The new B-ret point remains an external flag and its 80% interval includes zero. Scherer has nine people and 22 of 41 channels are interpolated. `niv-results/ba_six_fc_niv/report.json` SHA-256 `b150b1f8f771c95936bead19ac48f9e600ec42bc325e7efb793bd04978d434be`.

### RTX 2080 Ti compute projection

Each timing uses 1,131 measured windows, alternating arm order, fresh Solver per session and a discarded warm-up session. The projection adds the matched compute delta over cap250 to the sealed cap250 receipt. Model load is outside the projection. The first script also synchronized CUDA after each Team B and hard actor call.

| Profiled node | Job | cap250 ms/window | Fused nine25 projection (s) | Fused ba_six projection (s) |
|---|---:|---:|---:|---:|
| cn114 | 3147537 | 131.95 | 3,001.7 | 3,363.2 |
| cn116 | 3147538 | 138.27 | 3,061.5 | 3,376.5 |
| cn117 | 3147539 | 147.78 | 2,954.8 | 3,237.6 |
| cn122 | 3147540 | 129.96 | 3,054.1 | **3,483.5** |
| cn123 | 3147541 | 136.21 | 2,983.1 | 3,371.8 |

At 12:55 CET cn111 (3147536) was pending. All five measured cap250 baselines were below the exclusion threshold of 190.5 ms/window. Cn122 exceeded 3,450 s by 33.5 s and counts in the profiled result. Source `6aa00b22` repeats the exact ordinary Solver timing with component hooks disabled, jobs 3147645–3147650 on cn122, cn116, cn114, cn115, cn117 and cn123. The exact RTX PRO 4500 has not been measured.

The 08:08 CET Leonardo snapshot showed 26 running and 270 pending. This lane had 11 running and 187 pending. Codex heartbeat checks every 50 minutes and will retain the verified queue gaps in this card.
