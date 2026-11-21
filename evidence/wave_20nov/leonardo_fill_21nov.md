# Leonardo Team B · 21 November 2026

[Preregistration](../../docs/LEDGER.md#leonardo-fill--21-november-registered-before-71400-scoring) · [Dispatch](DISPATCH_21NOV.md)

## Finalists read before the new orders

The existing finalists head-to-head used 71200–71215. On the real folds, nine25 gained **+0.67 pp [+0.34, +0.97] held-new** and **+0.98 [+0.72, +1.25] returning** over hard. Nine15 gained +0.44 [+0.20, +0.65] and +0.76 [+0.51, +1.05]. Nine25 minus nine15 was +0.23 [+0.05, +0.41] and +0.22 [−0.03, +0.44]. `ba_terminal` minus nine25 was +0.00 [−0.20, +0.20] and +0.61 [+0.42, +0.80]. The existing `ba_terminal` ZIP exceeded the 3,600 s RTX 2080 Ti proxy on both nodes and triggered the Scherer returning flag. [ZIP qualification](final_ba_terminal.md)

S1 resolves the conditional seed lane. On the paired 71124–71131 replays, the nine25 gain was +0.97/+0.59 pp (seed 0), +0.95/+0.60 (seed 1), and +0.93/+0.80 (seed 2), held-new/returning. The three-seed mean was **+0.95 [+0.25, +1.53] / +0.67 [+0.36, +0.96]**. All three met the S1 robustness rule. Every paired seed difference included zero in both cohorts. The additional seed replay is unnecessary under the registered rule.

Leonardo collectors 65011278 and 65011279 completed. Their receipts are `cap250/runs/leonardo-fill-21nov/finalists-seeds.json` and `finalists-h2h.json`. Initial collector jobs 65011239 and 65011241 stopped before reading scores because the finalists release lacked the collector source. The retry staged the collector under this run and used the original finalists tapes.

## Fresh dose and returning comparison

Orders 71400–71431 are disjoint from the finalists replays. The first and second 16-order blocks will be reported alongside the pooled estimate. The instrument uses the same ten held-new people and P04/P10 returning people. Every interval is paired and resamples people and orders with official cell weights. The 57/43 proxy and 20/80 returning split are unverified.

Source `4a3d7872`. Smokes 65011374–65011376 passed v3 replay and matched all 1,695 pinned labels across the three arms on P01 visit 1, P04 visit 2 and P10 visit 2. The zero mixer was exact in the same call, both doses reached the live mixer, and the scorer watched for degraded terminal stages. At 08:08 CET the two 96-task arrays 65011605/65011606 were running or queued. Collectors 65011607, 65011608 and 65011609 wait for the first, second and both arrays respectively.

## Bank growth within +8 ms/window on A100

Source `be01260a`. The three arms are `swap25`, `fbc12` and `ba_six`. The first changes three Team B members at fixed count. The second adds three fitted FBC members. The third replaces three Team B members with the existing graph-based MANAS A1 terminal path. All four variants assembled from the real P01 banks locally, and the signed lead source reproduced byte-for-byte from its production bank. Growth live smokes 65011663–65011665 started at about 08:08 CET. They exercise both CUDA graph sizes, pinned lead labels, and the same-call zero and partner paths before any growth array.

The 08:08 CET Leonardo snapshot showed **26 running and 270 pending**, including Fable Round 3 and this lane. This lane had 11 running and 187 pending. Codex heartbeat is scheduled every 50 minutes from 08:57 CET in this thread to continue collection and dispatch through 18:00 CET.
