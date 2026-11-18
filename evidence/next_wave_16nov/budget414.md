# §414 Exact archive process and session timing

Cap250 used **1.67× and 1.84× incumbent process time** on the two four-session workloads. All 3,040 decisions matched their respective archived prefix outputs. Both preregistered mean timing predictions held. A private timeout does not follow from these measurements. Allocation and delivery order together changed absolute time enough to prevent a portable margin estimate.

| Timing in seconds | cn222 old → cap | cn205 cap → old |
|---|---:|---:|
| Whole process, incumbent / cap250 | 349.50 / 584.05 | 114.61 / 211.22 |
| Package import and load, incumbent / cap250 | 9.23 / 10.74 | 4.06 / 4.64 |
| Mean session copy/reset, incumbent / cap250 | 0.22 / 0.37 | 0.16 / 0.08 |
| First returning session, cap250 − incumbent | +22.55 | +7.72 |
| Warm returning session, cap250 − incumbent | +18.41 | +6.62 |
| First active session, cap250 − incumbent | +150.41 | +60.43 |
| Warm active session, cap250 − incumbent | +41.23 | +20.63 |
| First-active excess over warm active | +109.18 | +39.80 |

The first-active excess averaged 74.49 s against a 40–75 s prediction. The warm active delta averaged 30.93 s against a 15–40 s prediction. Both averages conceal substantial allocation spread. Each unknown prefix made 17 neural refits, 340 optimizer steps, nine personalization refits and 100 fused outputs. P10 was recognized at window 10 on every fresh copy and made no neural refits. All work counters repeated across allocations; no guard bound or stage reported an error. Mean package-load and copy/reset deltas were +1.04 s per process and +0.04 s per session, respectively. The delay concentrates inside prediction when an unknown recording first activates adaptation. [Guarded counts and paired timings](budget414_result.json)

| Process scenario | Standardized pairs whose additional cost fits 2,313 s, cn222 / cn205 | Pairs whose entire incumbent cost fits 1,287 s, cn222 / cn205 |
|---|---:|---:|
| One process | 36 / 83 | 8 / 29 |
| Two processes | 34 / 81 | 8 / 28 |
| Fresh process per pair | 13 / 33 | 6 / 17 |

Using average deltas gives 51, 49 and 18 pairs, respectively. On each allocation, the incumbent could execute fewer pairs in 1,287 s than the number needed to consume its observed timeout headroom. This supports retaining cap250 staged for this matched workload. These bounds assume additive measured costs for the standardized pair; they do not bound workloads with different session lengths, routes, hardware or process structure.

## Workload and evaluator

The pinned ingestion wrapper launches one `benchopt run -r 1` subprocess and records elapsed time around that invocation. A Benchopt solver loads the template model when setting its objective. The BCI objective enters one streaming loop over its test loader. On each session transition the loop releases the previous copy, deep-copies the template and calls its optional reset. The public warm-up specifies one dataset and one seed. The sealed phase configuration and process topology remain private. Its execution limit is 3,600 s. The sealed incumbent v5.1b recorded 1,287 s. [Ingestion](../../2026-competition/codabench/ingestion_program/ingestion.py) · [Solver](../../2026-competition/benchmark_utils/base_solver.py) · [Objective](../../2026-competition/tracks/bci_decoding/objective.py) · [Streaming loop](../../2026-competition/benchmark_utils/streaming.py) · [Phase limits](../../2026-competition/codabench/competition_bci.yaml) · [Historical result](../../docs/SEALED_UPLOADS.md)

Each archive started in a clean Apptainer process in the frozen official A100 worker image. The real 35-window smoke preceded both full allocation pairs. Each full process ran 190 windows from returning PROTEUS P10 day 3, 190 windows from each of two naturally unrecognized Schirrmeister recordings, then the identical P10 prefix on a fresh copy. The prefix rows and original labels came from §387. The two paired allocations reversed archive order and external recording order. A synchronized copy/reset and complete predictions were timed separately. Outer wall includes Apptainer, runtime imports, local array loading, model loading, predictions and shutdown. The official duration includes Benchopt and its private loader. Differences between these harnesses limit any extrapolation. The Schirrmeister tasks are external executed movements/rest. This workload measures runtime, with no new accuracy estimate. [Prior rows, archive identities and long streams](runtime387.md)

The scenario uses fixed 1,287 s historical runtime and 2,313 s headroom. Each additional standardized pair contains one returning and one naturally active 190-window session. Let W be the paired cap-minus-old time for one warm pair. Let P be the outer-process residual delta plus both first-session deltas minus W. A scenario with n pairs in p processes consumes nW + pP additional seconds. One- and two-process scenarios pay P once or twice; a separate process per pair pays nP. The residual includes package load and all time outside the four measured sessions. Floor the break-even calculation to whole pairs. The reference session counts and the cost of the private incumbent workload are unknown. As a consistency check, the corresponding incumbent timing alone bounds how many standardized pairs could fit inside 1,287 s on these workers, assuming zero additional competition overhead.

## Allocation history

The smoke (3124442) completed in 5m00s. The original cap→old pair (3124443_1, cn205) completed in 5m34s. The original old→cap pair (3124443_0, cn222) reached its 15-minute scheduling limit during cap250's last returning stream. The incumbent ran in 349.95 s there, versus 114.61 s on cn205. Its identical final P10 prefix alone took 71.86 s versus 21.73 s. Cap250 finished the first two external streams, each with 17 refits and 340 optimizer steps, and logged 106 and 9 s of fine-tuning. The censored cap250 process has no finished wall time. Job 3124585 repeated the complete old→cap pair on cn222 under a 25-minute limit. The rerun finished in 15m34s. Its incumbent took 349.50 s, within 0.45 s of the censored allocation’s incumbent. The paired analysis uses only complete allocations. Together with the censored pair and smoke, these jobs consumed 41m08s of allocated A100 time. All six recording-order v3 replay references passed locally.

Worker commit c15f202. Matched input and image hashes are published in §387. The worker reused the [original replay loader](../../studies/17nov/native_worker360/replay.py) with SHA-256 8b1dbe425d18fdb316e1528370d20cd78c0f5a9b4941246d2582effb39e93a10. Galvani directories ni26_budget414_v1 and ni26_budget414_ab_repeat contain the individual process timing records and source manifests.
