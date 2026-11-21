# Leonardo Team B · 21 November 2026

[Preregistration](../../docs/LEDGER.md#leonardo-fill--21-november-registered-before-71400-scoring) · [Dispatch](DISPATCH_21NOV.md)

## Finalists read before the new orders

The existing finalists head-to-head used 71200–71215. On the real folds, nine25 gained **+0.67 pp [+0.34, +0.97] held-new** and **+0.98 [+0.72, +1.25] returning** over hard. Nine15 gained +0.44 [+0.20, +0.65] and +0.76 [+0.51, +1.05]. Nine25 minus nine15 was +0.23 [+0.05, +0.41] and +0.22 [−0.03, +0.44]. `ba_terminal` minus nine25 was +0.00 [−0.20, +0.20] and +0.61 [+0.42, +0.80]. The existing `ba_terminal` ZIP exceeded the 3,600 s RTX 2080 Ti proxy on both nodes and triggered the Scherer returning flag. [ZIP qualification](final_ba_terminal.md)

S1 resolves the conditional seed lane. On the paired 71124–71131 replays, the nine25 gain was +0.97/+0.59 pp (seed 0), +0.95/+0.60 (seed 1), and +0.93/+0.80 (seed 2), held-new/returning. The three-seed mean was **+0.95 [+0.25, +1.53] / +0.67 [+0.36, +0.96]**. All three met the S1 robustness rule. Every paired seed difference included zero in both cohorts. The additional seed replay is unnecessary under the registered rule.

Leonardo collectors 65011278 and 65011279 completed. Their receipts are `cap250/runs/leonardo-fill-21nov/finalists-seeds.json` and `finalists-h2h.json`. Initial collector jobs 65011239 and 65011241 stopped before reading scores because the finalists release lacked the collector source. The retry staged the collector under this run and used the original finalists tapes.

## Fresh dose and returning comparison

Orders 71400–71431 are disjoint from the finalists replays. The instrument uses the same ten held-new people and P04/P10 returning people. Every interval is paired and resamples people and orders with official cell weights. The 57/43 proxy and 20/80 returning split are unverified.

Source `4a3d7872`. Smokes 65011374–65011376 passed v3 replay and matched all 1,695 pinned labels across the three arms on P01 visit 1, P04 visit 2 and P10 visit 2. The zero mixer was exact in the same call, both doses reached the live mixer, and the scorer watched for degraded terminal stages. At 08:08 CET the two 96-task arrays 65011605/65011606 were running or queued. Collectors 65011607, 65011608 and 65011609 wait for the first, second and both arrays respectively. The collector reproduced the historical Lane TB-verify gains for both cohorts and both doses from the original 71124–71131 predictions.

Both 96-task arrays and all three reports completed by 08:53 CET. The pooled `dose.json` SHA-256 is `c41c3f47f7eed1744605f298dc969df4c203f893a7345abd9633012e93aaf60b`.

| 32-order contrast | Held-new pp [80%] | Returning pp [80%] | P04 / P10 pp | J pp | 57/43* pp |
|---|---:|---:|---:|---:|---:|
| nine25 − hard | +0.87 [+0.45, +1.24] | +0.82 [+0.58, +1.04] | +0.66 / +0.98 | +0.85 | +0.89 |
| nine15 − hard | +0.66 [+0.42, +0.87] | +0.65 [+0.48, +0.81] | +0.58 / +0.72 | +0.65 | +0.67 |
| nine25 − nine15 | +0.21 [−0.01, +0.41] | +0.17 [+0.02, +0.30] | +0.08 / +0.26 | +0.19 | +0.22 |

Absolute BA by held-new / returning: hard 73.32% / 79.31%, nine15 73.98% / 79.96%, nine25 74.19% / 80.13%. Nine25 returning BA is 70.67% on P04 and 89.59% on P10. The two 16-order nine25 returning gains were +0.55 [+0.22, +0.85] and +1.09 [+0.84, +1.33]. Their 0.542 pp gap exceeds the registered 0.5 pp stability rule. The returning dose advantage passes the registered rule because its pooled lower bound is positive and both blocks favored nine25 (+0.06 and +0.28). Held-new dose advantage is undecided on these orders.

At 09:23 CET the queue had five running and one pending. I registered a second disjoint 32-order block (71600–71631) to check the returning order dependence and restore the queue. It uses the same folds, bank and inference path. Source `674930c0`, run `cap250/runs/leonardo-fill-retest-21nov`. The collector will report the new 32 and a pooled 64 with the original orders.

Retest smokes 65014371–65014373 passed all original pinned labels, zero and v3 checks. Leonardo accepted arrays 65014554/65014555 (96 tasks each) and collector 65014556. The 09:32 CET submission snapshot showed 25 running and 225 pending across the account, plus one completing.

## Bank growth within +8 ms/window on A100

Source `be01260a`. The three arms are `swap25`, `fbc12` and `ba_six`. The first changes three Team B members at fixed count. The second adds three fitted FBC members. The third uses six Team B members and the existing graph-based MANAS A1 terminal path. All four variants assembled from the real P01 banks locally, and the signed lead source reproduced byte-for-byte from its production bank. Growth live smokes 65011663–65011665 passed at 08:15 CET. They matched the lead and zero-control labels on 565 windows each, verified both CUDA graph sizes, and checked the exact MANAS terminal blend and session-local state.

At 08:15 CET controller `e8db6272` released nine remaining fold fixtures (65011888), six timings (65011889) and 48 screen tasks on 71500–71507 (65011890). Freeze 65011891 depends on the timings and the screen. Release 65011892 schedules held-out work on 71508–71523 only if a variant qualifies, then schedules its report. Slurm writes the IDs to `grow/screen-jobs.tsv` and `grow/held-jobs.tsv`.

| Screen vs nine25 | Held-new pp [80%] | Returning pp [80%] | P04 / P10 pp | J / 57/43* pp | Added A100 ms/window |
|---|---:|---:|---:|---:|---:|
| swap25 | +0.00 [−0.19, +0.17] | −0.27 [−0.53, −0.03] | −0.40 / −0.13 | −0.13 / −0.08 | −0.89 |
| fbc12 | −0.32 [−0.56, −0.09] | −0.34 [−0.52, −0.13] | −0.29 / −0.39 | −0.33 / −0.34 | +0.12 |
| ba_six | +0.03 [−0.25, +0.36] | +0.85 [+0.29, +1.52] | +1.19 / +0.51 | +0.44 / +0.29 | +2.23 |

Timing used 757 windows per arm across the four paired A100-SXM-64GB sessions. The `ba_six` pair deltas were −0.65, +2.56, +2.77 and +4.27 ms/window. The reference loaded first at both placements and its load time varied, so the negative load deltas are startup-order diagnostics. The decision uses the counterbalanced full-window times, including fresh-session setup. `selection.json` SHA-256 `f1c8cae836f681d9940c5e7eff03afb11f91efc3693556ab652172d7c0503ff0`.

The 96 held-out tasks (65013453) and report 65013454 completed by 09:26 CET. `ba_six` versus nine25 was **−0.10 [−0.39, +0.16] held-new and +0.70 [+0.41, +0.99] returning** across 16 disjoint orders. P04 +0.71, P10 +0.69, J +0.30 [80% +0.09, +0.50], and unverified 57/43 proxy +0.24. The held-new point and lower bound miss the promotion rule. The signed nine25 bank remains the lead from this lane.

## Second bank growth wave

Source `30074abb`, run `cap250/runs/leonardo-fill-roster-21nov`. The screen uses 71700–71715. `triseed25` spreads nine Team B members evenly across three existing fit seeds. `ba_seven` and `ba_eight` retain more of the original Team B bank and keep the same MANAS terminal term. All three assembled from the original trained P01 banks, with fresh signed CPU references. The same +8 ms/window A100 cap and growth retention rules apply. The confirmation orders are 71716–71731.

Smokes 65014561, 65014571 and 65014573 use P01, P04 and P10. Controller 65014651 waits for all three, then checks Slurm completion and the exact replay receipts before submitting the prep, timing and screen jobs. The 09:34 CET account snapshot had 33 running, 193 pending and ten configuring.

The 08:08 CET Leonardo snapshot showed **26 running and 270 pending**, including Fable Round 3 and this lane. This lane had 11 running and 187 pending. Codex heartbeat is scheduled every 50 minutes from 08:57 CET in this thread to continue collection and dispatch through 18:00 CET.
