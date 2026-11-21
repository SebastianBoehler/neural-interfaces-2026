# Leonardo Team B · 21 November 2026

**Lane recommendation at 13:56 CET.** Use `final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing.zip`, SHA-256 `890c2beec60344280f21b7206e99378dc71047a90b2e5db0876986b636b40cad`. The fresh fused fold contrast was +0.36 pp at 50/50. The worst of six normal RTX 2080 Ti nodes projected 3,259.2 s. The Scherer returning result remains a flag. The RTX PRO 4500 is unmeasured.

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

| 16 fresh fused-order contrast | Held-new pp [80%] | Returning pp [80%] | P04 / P10 pp | 50/50 pp [80%] |
|---|---:|---:|---:|---:|
| ba_six − nine25 | +0.21 [−0.04, +0.46] | +0.50 [+0.26, +0.74] | +0.47 / +0.53 | +0.36 [+0.17, +0.54] |

All 96 replays and the collector completed by the 13:36 CET check. The returning lower bound and positive held-new point meet the user’s revised quality rule. The earlier disjoint block gave 50/50 +0.30 [+0.09, +0.50]. Both blocks reuse people. `confirmation.json` SHA-256 `d96335c7b18ae1d07b5b95ac50883fb2c76348dbdd19b712c6a0ebdccc9f2bca`.

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

| Unprofiled node | Job | cap250 ms/window | Fused nine25 projection (s) | Fused ba_six projection (s) |
|---|---:|---:|---:|---:|
| cn114 | 3147647 | 135.73 | 3,042.0 | **3,468.3** |
| cn115 | 3147648 | 129.42 | 3,088.1 | 3,412.1 |
| cn116 | 3147646 | 131.22 | 3,112.1 | 3,334.0 |
| cn117 | 3147649 | 135.10 | 2,963.1 | 3,355.3 |
| cn122 | 3147645 | 145.03 | 2,850.2 | 3,191.7 |
| cn123 | 3147650 | 135.32 | 3,024.1 | 3,371.1 |

All six unprofiled runs completed by 13:12 CET, and all six cap250 baselines qualify as normal. Cn114 misses the 3,450 s target by 18.3 s. On cn122 the candidate measured 176.20 ms/window with hooks and 176.73 without. Cap250 changed from 129.96 to 145.03 ms/window. The cn122 projection change came from the different cap250 baseline. The exact fused `ba_six` ZIP is timing-ineligible under the user's current rule. Signed fused nine25 is the timing-safe choice from this lane.

### Turing-only MANAS acceleration

Registered one targeted attempt after the timing miss. On CUDA capability 7.5, where BF16 lacks native arithmetic, test FP16 autocast in the MANAS encoder. The CPU and BF16 paths keep their original operators. The patch copies all weights and changes only generated Solver source. The first draft could have treated BF16 emulation as native support. Superseded build 3147734 and canceled dependent smoke 3147735. Source `33993df2` selects 7.5 explicitly and observes the recorded dtype at graph capture. Revised build 3147737 and paired 379-window smoke 3147738 run on Galvani. Any changed active labels or stage activity blocks the variant.

Smoke 3147738 passed at 13:22 CET. The parent used BF16 emulation on cn114 and the patched encoder used FP16. The 379 labels and stage counts matched. The largest absolute traced score difference was 1.09e−4. Production SHA-256 `890c2beec60344280f21b7206e99378dc71047a90b2e5db0876986b636b40cad`. Scherer-free SHA-256 `0804a2f272d4fb4b6215cfcd7f282f0c813cbe4b4888f3f6b5f38f74964d6160`.

Source `19f9b979` released seven all-public identity shards 3147754, CPU zero 3147755, signed-lead official pair 3147756, and six unprofiled timing jobs 3147757–3147762. The nodes in job order are cn114, cn116, cn115, cn117, cn122 and cn123. The timing pairs both `ba_six` archives, cap250 and signed nine25_fc. Paired NI-V smokes 3147763/64 gate controller 3147765. A100 BF16 identity check is 3147766.

All-public shards confirmed 0/2,655 changed labels across all 14 sessions, with identical stage counts. The largest absolute traced score difference on 2080 Ti was 5.02e−4. The A100 check observed BF16 on both archives and bitwise identity on 379 windows. Official fused-lead pair 3147756 passed with stable scores and unchanged files. Lead 99.5923%, FP16 candidate 100.0000%.

Controller 3147765 released 36 paired B-new tasks 3147783, nine paired B-ret tasks 3147784 and report 3147785 over the same eight Scherer orders as the ordinary variant. The exact Galvani FP16 ZIP is mirrored locally and SHA-verified. Native-BF16 e2e check 3147847 pairs both signed variants on A100.

CPU zero 3147755 passed both 379-window checks at 13:41 CET, with Team A disabled and with both Team A and Team B disabled. All six ordinary Solver timings completed with 1,131 measured windows per arm and normal cap250 baselines.

| Node | Job | cap250 ms/window | Fused nine25 projection (s) | Original ba_six projection (s) | FP16 ba_six projection (s) | Paired FP16 saving (ms/window) |
|---|---:|---:|---:|---:|---:|---:|
| cn114 | 3147757 | 135.22 | 3,109.8 | 3,372.4 | 3,100.8 | 13.53 |
| cn115 | 3147759 | 138.47 | 2,872.6 | 3,242.0 | 3,009.5 | 11.58 |
| cn116 | 3147758 | 134.02 | 3,102.1 | 3,488.4 | **3,259.2** | 11.41 |
| cn117 | 3147760 | 148.35 | 2,903.8 | 3,225.8 | 3,040.3 | 9.24 |
| cn122 | 3147761 | 135.75 | 3,026.6 | 3,452.4 | 3,151.5 | 14.98 |
| cn123 | 3147762 | 144.60 | 3,099.0 | 3,462.9 | 3,200.6 | 13.06 |

The patched archive passes the worst-normal-node timing rule with **190.8 s** to the 3,450 s target. Every measured node counts. Compute projections exclude initial load, use matched cap250 and retain full session startup and capture. Scherer and the supplemental native-BF16 timing were pending at the 13:41 CET snapshot.

All 45 paired Scherer tasks and collector 3147785 completed by 13:49 CET. The candidate’s B-ret full-session point remains negative. Its paired 80% interval includes zero. The original full MANAS terminal’s B-ret flag was −0.52 pp [−1.01, −0.02].

| FP16 ba_six vs fused nine25 delivery | B-new pp [80%] | B-ret pp [80%] | 50/50 pp |
|---|---:|---:|---:|
| Full-session | +0.09 [−0.41, +0.56] | −0.41 [−1.09, +0.22] | −0.16 [−0.56, +0.24] |
| Within | −0.01 [−0.34, +0.31] | −0.19 [−0.74, +0.36] | −0.10 |

Worst B-ret person remained −3.85 pp. `niv-results/ba_six_fc_niv_fp16_turing/report.json` SHA-256 `27d99ded4bb74c10236fd98d109572429ecebf44c0748d93f114ee31a8fa4a7d`. The user’s rule retains this as an external flag.

Native BF16 timing 3147847 finished at 13:54 CET on A100-PCIe-40GB cn207. Cap250, signed nine25, original `ba_six` and the FP16-on-Turing archive used 127.65, 152.39, 157.14 and 156.93 ms/window respectively. The archive-to-parent gap was **−0.21 ms/window**, within the registered ±2 ms expectation. Projected compute 3,142.9 s. The A100 parity job separately verified that both use BF16 there.

### Queue receipt

Leonardo had 26 running and 270 pending at 08:08 CET. It dropped below 15 by 09:23 and both own-job queues were empty at the 12:11–12:15 check. The afternoon waves restored the queues. At 13:56 CET Leonardo had 34 running and 152 pending, and Galvani had one running and 20 pending. Galvani job 3147536 was an extra original `ba_six` timing on cn111. The FP16 release decision has six measured normal nodes. The Codex heartbeat is scheduled through the 18:00 CET cutoff.

14:01 CET heartbeat. Leonardo had 39 running and 115 pending tasks across the account, with the afternoon fold waves queued for 14:45 and 16:25 CET. The dose, retest, roster and fused confirmation arrays and collectors were complete. Local SHA-256 checks of all four copied receipts matched the card. The 71800–71815 collector retained `ba_six` at +0.3566 pp (50/50). The queue exceeded the 15-task target at this check.

14:07 CET heartbeat. Leonardo reported 11 running and 98 pending tasks after an initial SSH port-22 timeout. Galvani reported 11 running and 67 pending at 14:06 CET. The three Galvani repeat smokes completed. The first FP16 timings were queued on priority, with the second repeats and Scherer collector waiting on dependencies. Original fused `ba_six` timing 3147536 completed at 14:02 CET on RTX 2080 Ti cn111 over 1,131 windows per arm. Cap250 was normal at 140.17 ms/window. Fused nine25 projected 2,955.5 s and original `ba_six` projected 3,289.3 s. The recorded misses on other normal nodes still govern the original archive. Receipt `out/e2e-3147536.json` SHA-256 `46c3670aaa1b4d6fd04ea918a7455802a59615bd70eb2c63cfe1712adcdd9102`.

### 15:00 CET follow-on receipts

At 15:00 CET Leonardo showed 11 running and 134 pending tasks across the account, including 10 running and 54 pending in the 71900–71915 wave. Collector 65024442 was pending and the final four orders remain eligible from 16:25 CET. Galvani showed five running and 22 pending across the account. The three first FP16 repeats on cn116, cn122 and cn123 waited for resources. The six second repeats are eligible from 16:55 CET after their first repeats. Both queues contained active work. [Job IDs](OVERNIGHT_21NOV.md#10-galvani-fp16-decision-repeat)

Galvani Scherer collector 3147914 completed at 14:51:56 CET for fresh orders 71848–71855. The exact FP16 Scherer-free arm is compared with signed fused nine25. The same nine people were used in the first block.

| Delivery | B-new pp [80%] | B-ret pp [80%] | 50/50 pp [80%] |
|---|---:|---:|---:|
| Full-session | +0.38 [−0.10, +0.84] | +0.18 [−0.53, +0.88] | +0.28 [−0.14, +0.71] |
| Within-run | −0.20 [−0.64, +0.23] | −0.38 [−0.97, +0.21] | −0.29 (cohort point mean) |

The full-session returning point was positive, with a worst-person loss of 1.88 pp. Its interval includes zero and the forecast point. The earlier −0.41 pp full-session B-ret flag still applies. The repeat's secondary within-run B-ret point also falls below −0.30 pp. `niv-results/ba_six_fc_niv_fp16_turing/report.json` SHA-256 `00daf05768a706aff99abbb29d566f512175899d154d9fd7b803b7ef083d7e05`. The full-session 50/50 point and interval are from the collector's paired draws.

The first three warm unprofiled Galvani FP16 repeats completed on RTX 2080 Ti. All used 1,131 measured windows per arm. All cap250 baselines were normal under the rule excluding values above 190.5 ms/window. Candidate model load was 5.05–5.53 s and is outside the compute projection.

| Node | Job | Cap250 ms/window | FP16 projected s | `out/timing-<job>.json` SHA-256 |
|---|---:|---:|---:|---|
| cn111 | 3147915 | 136.45 | 3,278.8 | `3e4e7c471f56c9b24d0b25703eaabeb0266a00e1b40d9e4a3848c426eeadb449` |
| cn114 | 3147917 | 149.44 | 3,246.9 | `138d6761adf681cbc497af1e94a4b322b04e320b27b6d19b7d9711910c9db26f` |
| cn115 | 3147919 | 133.67 | 3,280.7 | `784d4ff6f8ad66db704716bbd7d1ae690923e99e567a4fa4e06c9b30ce5da020` |

The largest screened FP16 projection is now 3,280.7 s, leaving 169.3 s to the 3,450 s target. The shipping recommendation still qualifies. Nine repeat receipts remain pending. [Repeat protocol](OVERNIGHT_21NOV.md#10-galvani-fp16-decision-repeat)

Leonardo afternoon job 65024312 ran the separately registered A100-SXM comparison of fused nine25 and the 10% Team A terminal archive over 1,131 warm windows per arm. The terminal archive's median was 2,995.3 s, below its 3,150–3,600 s forecast band. Its median paired cost over nine25 was +4.60 ms/window, below the forecast of at least +10. Model load is outside both projections. These are A100-SXM receipts for the archives in the afternoon protocol. [Protocol](OVERNIGHT_21NOV.md#9-leonardo-afternoon-confirmation)

| Node | Rank | Cap250 ms/window | Nine25 projected s | 10% Team A terminal projected s | `out/e2e-65024312-<rank>.json` SHA-256 |
|---|---:|---:|---:|---:|---|
| lrdn3302 | 0 | 116.00 | 2,902.9 | 2,995.3 | `3e1c071676909d3620860bcf882819626d6d0edd19ff248e3268b067f20a41ba` |
| lrdn3362 | 1 | 114.97 | 2,910.3 | 2,992.2 | `41a5180a71e85b7382387e393b0f211214ee1ec6a2e422454f6a5a8830722a47` |
| lrdn3385 | 2 | 117.24 | 2,911.2 | 3,004.2 | `5f88a57b49d1a5b0559c2bf7e135519b1954d9c6267f51c843c25bd9a4246348` |

### 15:47 CET queue receipt

Leonardo had one running and 49 pending across the account. All 48 lane tasks for 71908–71911 completed with exit 0:0. The lane had a running gap while 48 tasks for 71912–71915 waited for the registered 16:25 CET release. Collector 65024442 still waits for that wave. The running-plus-pending count stayed above 15 at this check.

Galvani had zero running and 20 pending across the account, including all nine outstanding FP16 timings. The first cn116, cn122 and cn123 repeats (3147921/23/25) were pending for resources. Slurm's 15:47 snapshot estimated 02:01 CET on 22 Nov for all three, subject to placement changes. Their second repeats wait on those jobs. The other three second repeats remain eligible from 16:55 CET. No new timing or Scherer receipt appeared. The worst observed normal FP16 projection remains 3,280.7 s. Existing jobs cover the registered checks.
