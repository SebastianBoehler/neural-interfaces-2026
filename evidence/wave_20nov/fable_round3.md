# Fable Round 3 · 21 November 2026

[Preregistration](../../docs/LEDGER.md#fable-round-3--21-november-registered-before-scoring) · [Dispatch log](DISPATCH_21NOV.md) · [Scorers](../../scripts/fable_round3/)

Signed reference `final_stack_hard_safe_st_team_b_nine25_tbgraph.zip`, SHA-256 `6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9`. The protocol uses 28 held-new cells across ten people and eight returning cells across P04/P10. Fit orders 71300–71307 and held-out orders 71308–71315 shuffle the full session. The order blocks reuse the same people.

## Verification before scoring

- The local `replay/check.py` reproduced all six v3 reference scores and printed `OK`. The scorer matched the official balanced accuracy for each public test cell to 1e−12 and returned an exact zero on a paired diagnostic.
- Galvani A100 smoke 3146351 completed in 86 s. It exercised the signed lead on 72 public windows, verified native and cached Team B label parity, tapped pre-EM and terminal scores, applied the live rebalance and verified prefix and frozen query-order parity. The public smoke did not measure accuracy.
- Exact fold, Team B weight and tape inputs are on Leonardo. Batch SSH returned `Permission denied` at 06:59, 07:09 and 07:19 CET. Decisive scoring awaits the pinned-fold smoke.
- Galvani jobs 3140818 and 3140819 are `CANCELLED`. A full queue scan found the latter was the only child of the former and found no other dependents.
- Public smoke 3146352 completed in 86 s from `cabc1e92`. Bet 2's zero and unit weights reproduced the hard-base and Team B labels exactly.
- State-only smoke 3146353 completed in 53 s on Galvani. The signed lead processed 100 unrecognized Scherer visit-1 windows with eight fine-tuning refits and four CSP refits. The frozen checkpoint reproduced its labels when the query order was reversed. The worker read waveform tensors and session identifiers for this smoke.
- Source-only smoke 3146356 checked both prospective changes against the scorer on 72 signed-lead windows. Bet 1 at 25% from window 15 and Bet 2 at weights 0.20/0.45 reproduced the scorer's labels, and both disabled controls reproduced the lead labels. The smoke used unpacked source in a temporary directory.

Leonardo access returned at 07:39 CET. Exact input paths were verified and the pinned-fold smokes and dependent waves were submitted under `cap250/runs/fable-round3-21nov` from `f4d3714a`. [Slurm job IDs](DISPATCH_21NOV.md)

At 07:48 CET, Leonardo jobs 65010877 and 65010878 had both completed. Each printed v3 `OK`. They matched the hard-only and nine25 pinned order 71124 labels on P01 visit 1 (190 windows) and P04 visit 2 (189 windows). The fit and held-out trace arrays started after both smokes.

## Fit screen on orders 71300–71307

All 96 trace parts completed. Screen 65010881 finished at 08:06 CET. Its [receipt](fable_round3_screen.json) has SHA-256 `7ac1472f61bbad6e0fcd76cc383f8b203d24ee950c0df89e6a0cf288f74755c3`.

- Step 0 frozen gain was **+2.32 [+1.38, +3.36] pp** held-new and **+0.98 [+0.34, +1.61] pp** returning. The returning lead gained 6.77 [2.87, 10.24] pp from windows 1–20 to windows 41+. The two preregistered conditions released Bet 3.
- Bet 1 Team B rebalance improved first-15-window accuracy by +0.07 [−1.64, +1.64] pp held-new and +1.67 [−1.03, +4.36] pp returning. The fit rule kept the rebalance active from window 1. The base settings are alpha 1.15, prior mass 5 and a 30-window history.
- Bet 2 hard-final accuracy dropped 26.38 [22.52, 29.52] pp held-new and 30.92 [14.46, 55.82] pp returning on changed windows. Team B trailed hard-final there by **−2.88 [−12.89, +3.97] pp** and **−3.74 [−15.59, +27.07] pp**, respectively. The registered stop rule rejected fitting a weight pair. These are conditional accuracy contrasts.

| Cohort and adaptation set | Hard-final % | Hard pre-EM % | Team B % |
|---|---:|---:|---:|
| Held-new, unchanged | 76.35 | 76.35 | 67.19 |
| Held-new, changed | 49.98 | 36.91 | 47.09 |
| Returning, unchanged | 83.25 | 83.25 | 76.12 |
| Returning, changed | 52.33 | 35.39 | 48.59 |

## Held-out ceiling and Team B diagnostics on orders 71308–71315

Frozen minus causal was **+2.09 [+1.25, +3.18] pp** held-new and **+1.04 [+0.33, +1.77] pp** returning. Each frozen query starts from the same final session checkpoint and retains the lead's current-window normalization. The intervals resample people and delivery orders.

| Delivery position | Held-new causal % | Held-new frozen % | Held-new gain, pp [80%] | Returning causal % | Returning frozen % | Returning gain, pp [80%] |
|---|---:|---:|---:|---:|---:|---:|
| 1–10 | 67.85 | 76.36 | +8.50 [+4.95, +12.57] | 73.65 | 81.48 | +7.83 [+3.72, +11.91] |
| 11–20 | 70.15 | 77.27 | +7.12 [+4.31, +10.35] | 81.15 | 84.93 | +3.78 [+0.74, +7.26] |
| 21–40 | 71.54 | 76.08 | +4.54 [+2.57, +7.13] | 81.00 | 83.52 | +2.53 [-0.02, +5.18] |
| 41+ | 75.67 | 76.66 | +0.99 [+0.34, +1.81] | 80.83 | 81.04 | +0.21 [-0.41, +0.83] |

Held-out 41+ minus 1–20 was +6.67 [+3.80, +9.58] pp held-new and +3.43 [-0.99, +7.53] pp returning. The returning point fell below the fit block's 5 pp release threshold.

Team B alone moved from 65.28% to 64.90% held-new (-0.37 [-0.81, +0.02] pp) and from 72.79% to 73.17% returning (+0.38 [-0.41, +1.13] pp). Local collection reconstructed all 48 held-out trace parts and matched the fit block's fold receipts. SHA-256 `18463bcb7f79406ba599f58863fc4655c924c93dd2310f5e8390417f6930c40d`.

## Bet 3 · enrolled covariance reference

The two anchor smokes passed pinned native zero. The returning smoke applied all three nonzero doses after P04 recognition and preserved every earlier label. Slurm completed 12 reference banks, eight fit chunks, and 48 held-out chunks. The enrolled references use the latest visit that each fold permits.

| Pseudo-count | Returning fit delta, pp [80%] |
|---:|---:|
| 4 | −0.13 [−0.34, +0.07] |
| **8** | **+0.10 [−0.02, +0.23]** |
| 16 | −0.12 [−0.30, +0.06] |

The frozen dose of 8 gave **+0.005 [0.000, +0.012] pp held-new** and **−0.167 [−0.335, −0.002] pp returning**. P04 was −0.24 pp and P10 −0.09 pp. J was −0.08 pp [−0.16, +0.00]. The unverified 57/43 proxy was −0.05 pp. Returning windows 1–20 changed by −0.46 [−1.23, +0.32] pp. Bet 3 missed the returning gate.

The anchor activated after 11–49 deliveries on all 32 returning streams. It also activated once in 112 held-new streams when the native gate recognized P10 visit 1 as P03 on order 71309. The first eligible prediction follows the recognition delivery. The other 111 held-new streams reproduced the lead exactly.

Fit selection 65011779 completed at 08:32 CET with SHA-256 `ed55c8eee0a060644b7136aa7549e3b2a8cf10693b3ee9c9505694a936736be0`. Held-out collector 65011781 completed at 08:44 CET with SHA-256 `f2c93579fbfae08b4937ff04dac391b64c67f892100fc81c873db57659896803`.
