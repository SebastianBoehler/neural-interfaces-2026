# §365 — Causal personalization refit scheduling

The frozen v6.6cap250 terminal personalization stage fits its models at windows 20, 40 and 60. Each is replaced before the fusion weight becomes nonzero at window 90. The public replay checked whether we could skip those three fits or spend their work later in each session.

| Schedule | Refit windows through 180 | Calls | Prefix rows per session |
|---|---|---:|---:|
| Frozen | 20, 40, 60, 80, 100, 120, 140, 160, 180 | 9 | 900 |
| Skip | 80, 100, 120, 140, 160, 180 | 6 | 780 |
| Row-budget | Skip plus 110 | 7 | 890 |
| Recycle | Skip plus 90, 110 and 130 | 9 | 1,110 |

Only the schedule predicate changes. Every fit uses windows strictly earlier than its refit index. All arms reuse the frozen B2_pool upstream bank, count filter, original personalization fit and fusion, and exact order file. The frozen checkpoints and original auxiliary exposure are inherited. No new cross-person fitting occurs. Package and source checks, archived control parity, active refit budgets and session/prefix/recognition guards run before scoring.

The 3122635 no-score smoke completed in 114 s. It reproduced frozen and skip bitwise across all 2,655 windows in within0. The warmed reverse order also matched. Recognized sessions in the earlier paired archive stopped by index 63, before index 90. Under that archived behavior the schedule changes no recognized-returning output. Source commit `08dc7c1`.

## Results

Job 3122636 completed in 342 s. The replay used all eight shuffle orders, 2,655 windows and 14 sessions per order. Frozen and skip scores matched bitwise in every order. All schedules made the same 11,160 fusions. Accuracy is official equal-cell new-person B2_pool public DEV. Intervals resample people and order.

| Schedule | Accuracy | Official delta pp, 80% interval | People/order delta pp, 80% interval |
|---|---:|---:|---:|
| Frozen | 76.27934% | | |
| Skip | 76.27934% | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] |
| Row-budget | 76.29940% | +0.02006 [−0.01734, +0.05787] | +0.02442 [−0.01292, +0.06232] |
| Recycle | 76.28953% | +0.01020 [−0.04467, +0.07339] | +0.03429 [−0.02211, +0.09475] |

Row-budget changed 1–5 labels per order and helped 6/10 people in the scheme-balanced comparison. Its P04/P10 deltas were +0.06173/−0.04340 pp. Recycle changed 3–8 labels per order and helped 6/10. Its P04/P10 deltas were −0.05495/−0.04510 pp. Both accuracy effects remain undecided under the registered rule.

| Schedule | Refit calls | Prefix rows | Selected rows | Time in refits | Cached CP wall |
|---|---:|---:|---:|---:|---:|
| Frozen | 1,008 | 100,800 | 51,208 | 37.341 s | 43.798 s |
| Skip | 672 | 87,360 | 44,179 | 29.261 s | 35.721 s |
| Row-budget | 784 | 99,680 | 50,414 | 33.627 s | 40.114 s |
| Recycle | 1,008 | 124,320 | 62,903 | 42.333 s | 48.833 s |

The exact skip removed one-third of the refit calls and saved 18.44% cached CP wall time and 21.64% measured refit time. Recycle used 23.33% more prefix rows than frozen at the same call count and added 11.50% cached CP wall time. The wall timer covers only the personalization loop with immutable, shared per-window covariance features. Worker timing and long-stream parity need separate qualification. Baseline early-fit errors can change the stage's later behavior, so this exactness result covers the tested error-free paths.

The two jobs consumed 912 CPU core-seconds. Source manifest `9ceab10901adfaebdfbd11644034abeb931f2e51811f06d73a1a3131f1e9bc36`. Pre-score proof `65078750a7e8737e0acf58f6a8831eb708fdd43f1ca26b91a5c9473d19c3817b`. [Result JSON](cp_schedule365_result.json) SHA-256 `f4e8286abcd76bc99d1df58c83b43736cff2d85fb73e288a0dd998c492ecb71a`.
