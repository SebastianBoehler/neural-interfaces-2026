# Track 2 bottleneck review · 21 November 2026

**Interim, 14:29 CET.** Across 16 pinned orders, ba_six reaches **77.68%** at 50/50 against **77.32%** for nine25. P04’s four returning cells cost **7.11 pp** of ba_six’s **22.32 pp** error budget. Held-new word generation costs **4.89 pp**, the largest class loss in that cohort. Deliveries 41+ contribute **16.97 pp** of total loss.

## Scope and checks

The instrument uses real PROTEUS fold weights from `production-20261120-32a94ab7`. The 28 held-new cells span ten people. The eight returning cells span P04 and P10 on visits 2 and 3, with visit 1 enrolled. P04 and P10 also appear in the held-new instrument with different fit exclusions. Cell means weight true classes equally. Each held-new cell carries 1/56 of the joint score and each returning cell 1/16. Tables use `new` and `ret` for these cohorts. The two-person returning instrument limits cohort generalization. [State](../../docs/STATE.md) · [Leakage audit](redteam_fable.md)

Both label banks from Leonardo `leonardo-fill-ba-six-21nov/grow/held` matched every label and delivery row in the independent `last-push-21nov/{fit,held}` zero arms and the regenerated full-session shuffle. The pinned v3 replay printed `OK`. Shuffled error analysis independently reproduces the +0.3566 pp comparison and its interval. The ba_six A100 fold path uses native BF16. The signed FP16-on-Turing archive follows that path on A100, with a separate 379-window bitwise A100 identity check and a 2,655-window zero-label-change Turing check on the all-data package. [Fold confirmation and precision checks](leonardo_fill_21nov.md)

The current fold tapes retain terminal decisions. Galvani container smokes 3148143/3148144 changed 3/190 nine25 labels and 1/190 ba_six labels on P01. Native Galvani smokes 3148202–3148205 repeated those counts on P01 and changed 3/189 and 2/189 labels on returning P04. Leonardo’s original environment passed all four smokes (65032096/65032104/65032107/65032110), covering 1,134 pinned labels, exact same-tape replay and 32-window prefix checks. The 96 capture tasks in 65032114 use the same frozen fit assets and orders 71800–71803 / 71808–71811. [Preregistration](../../docs/LEDGER.md#bottleneck-review--21-november-registered-before-analysis)

## Error budget

| Stack | Held-new | Returning | 50/50 BA | New lost pp | Ret. lost pp |
| --- | --- | --- | --- | --- | --- |
| nine25 | 74.26 | 80.38 | 77.32 | 12.87 | 9.81 |
| ba_six | 74.48 | 80.88 | 77.68 | 12.76 | 9.56 |

Paired ba_six gains in held-new / returning / 50/50 order were +0.21 [-0.04, +0.46] · +0.50 [+0.26, +0.74] · +0.36 [+0.17, +0.54] pp (80% intervals).

### People

| Person | New nine25 | New ba_six | Ret. nine25 | Ret. ba_six |
| --- | --- | --- | --- | --- |
| P01 | 72.44 | 73.72 | — | — |
| P02 | 86.70 | 87.19 | — | — |
| P03 | 72.25 | 72.72 | — | — |
| P04 | 52.40 | 52.52 | 71.08 | 71.55 |
| P05 | 75.36 | 74.24 | — | — |
| P06 | 83.17 | 83.69 | — | — |
| P07 | 87.53 | 87.79 | — | — |
| P08 | 85.81 | 86.42 | — | — |
| P09 | 80.54 | 80.04 | — | — |
| P10 | 79.57 | 79.77 | 89.68 | 90.21 |

### Sessions and cells

Visit BA averages the two interface cells. Each interface cell shows window count · nine25/ba_six BA% · nine25/ba_six lost 50/50 pp.

| Cohort | Person | Visit | Visit BA% | BrainHero | Graz |
| --- | --- | --- | --- | --- | --- |
| new | P01 | 1 | 72.4/73.7 | 94 · 81.0/82.1 · 0.34/0.32 | 96 · 63.9/65.3 · 0.65/0.62 |
| new | P02 | 1 | 86.7/87.2 | 93 · 87.4/87.0 · 0.22/0.23 | 96 · 86.0/87.4 · 0.25/0.23 |
| new | P03 | 1 | 72.3/72.7 | 93 · 68.4/68.7 · 0.56/0.56 | 96 · 76.1/76.8 · 0.43/0.42 |
| new | P04 | 1 | 56.9/56.2 | 93 · 61.4/60.4 · 0.69/0.71 | 96 · 52.3/52.0 · 0.85/0.86 |
| new | P04 | 2 | 50.4/50.7 | 93 · 51.4/50.5 · 0.87/0.88 | 96 · 49.3/50.8 · 0.90/0.88 |
| new | P04 | 3 | 50.0/50.7 | 92 · 56.0/56.3 · 0.79/0.78 | 96 · 43.9/45.1 · 1.00/0.98 |
| new | P05 | 1 | 75.4/74.2 | 94 · 76.8/76.1 · 0.41/0.43 | 96 · 73.9/72.4 · 0.47/0.49 |
| new | P06 | 1 | 83.2/83.7 | 91 · 80.2/80.4 · 0.35/0.35 | 96 · 86.1/87.0 · 0.25/0.23 |
| new | P07 | 1 | 87.5/87.8 | 93 · 90.4/90.8 · 0.17/0.17 | 96 · 84.6/84.8 · 0.27/0.27 |
| new | P08 | 1 | 85.8/86.4 | 93 · 78.7/79.5 · 0.38/0.37 | 96 · 92.9/93.4 · 0.13/0.12 |
| new | P09 | 1 | 80.5/80.0 | 93 · 79.3/78.8 · 0.37/0.38 | 96 · 81.8/81.2 · 0.33/0.33 |
| new | P10 | 1 | 68.3/69.7 | 96 · 65.9/67.8 · 0.61/0.58 | 96 · 70.7/71.7 · 0.52/0.51 |
| new | P10 | 2 | 86.9/86.0 | 90 · 92.2/91.9 · 0.14/0.15 | 96 · 81.7/80.1 · 0.33/0.35 |
| new | P10 | 3 | 83.5/83.6 | 103 · 86.9/86.6 · 0.23/0.24 | 96 · 80.1/80.5 · 0.36/0.35 |
| ret | P04 | 2 | 69.2/70.2 | 93 · 72.1/72.9 · 1.74/1.69 | 96 · 66.3/67.5 · 2.11/2.03 |
| ret | P04 | 3 | 73.0/72.9 | 92 · 76.1/76.3 · 1.50/1.48 | 96 · 69.9/69.5 · 1.88/1.90 |
| ret | P10 | 2 | 88.0/88.9 | 90 · 94.7/95.6 · 0.33/0.28 | 96 · 81.4/82.2 · 1.16/1.11 |
| ret | P10 | 3 | 91.3/91.5 | 103 · 93.9/94.2 · 0.38/0.36 | 96 · 88.8/88.8 · 0.70/0.70 |

The five largest ba_six cell losses were P04/2 Graz (ret, 2.03 pp), P04/3 Graz (ret, 1.90 pp), P04/2 BrainHero (ret, 1.69 pp), P04/3 BrainHero (ret, 1.48 pp), and P10/2 Graz (ret, 1.11 pp).

Across the same test visits, the enrolled fold improves ba_six by 20.87 pp for P04 (50.68 → 71.55%) and by 5.41 pp for P10 (84.80 → 90.21%). P04 returning recalls on Graz are 69.53% imagery, 64.75% subtraction and 71.29% word generation. On BrainHero its word recall is 69.56%, and imagery ↔ word accounts for 45.51% of its errors. This comparison changes both fit enrollment and the live routing path.

### Interface and position

Each model shows BA% / lost 50/50 pp. Position bins reuse full-cell class weights and renormalize within the bin for accuracy. Losses add across bins.

| Cohort | Stratum | nine25 | ba_six |
| --- | --- | --- | --- |
| new | BrainHero | 75.43 / 6.14 | 75.49 / 6.13 |
| new | Graz | 73.10 / 6.73 | 73.47 / 6.63 |
| new | 1–10 | 67.90 / 0.85 | 68.17 / 0.84 |
| new | 11–20 | 67.99 / 0.84 | 68.04 / 0.84 |
| new | 21–40 | 72.76 / 1.44 | 73.55 / 1.40 |
| new | 41+ | 75.31 / 9.74 | 75.46 / 9.68 |
| ret | BrainHero | 84.18 / 3.95 | 84.74 / 3.81 |
| ret | Graz | 76.58 / 5.86 | 77.02 / 5.75 |
| ret | 1–10 | 71.20 / 0.76 | 71.78 / 0.74 |
| ret | 11–20 | 77.91 / 0.58 | 79.32 / 0.54 |
| ret | 21–40 | 80.60 / 1.02 | 81.32 / 0.98 |
| ret | 41+ | 81.13 / 7.45 | 81.53 / 7.29 |

### Confusion

Rows show true-class recall distributions. Columns within each triple are 0 motor imagery, 1 mental subtraction, 2 word generation. The final column gives the largest bidirectional pair and its share of off-diagonal mass.

| Cohort | Interface | Stack | True 0 | True 1 | True 2 | Top pair |
| --- | --- | --- | --- | --- | --- | --- |
| new | BrainHero | nine25 | 79.5 / 8.6 / 12.0 | 9.0 / 75.4 / 15.6 | 15.0 / 13.5 / 71.5 | 1↔2 · 39.5% |
| new | BrainHero | ba_six | 79.2 / 8.3 / 12.5 | 8.2 / 75.5 / 16.3 | 15.4 / 12.8 / 71.8 | 1↔2 · 39.6% |
| new | Graz | nine25 | 77.3 / 11.7 / 11.0 | 12.8 / 72.3 / 14.9 | 16.1 / 14.2 / 69.7 | 1↔2 · 36.1% |
| new | Graz | ba_six | 77.8 / 11.2 / 11.0 | 12.6 / 73.0 / 14.4 | 16.5 / 13.9 / 69.5 | 1↔2 · 35.6% |
| ret | BrainHero | nine25 | 84.4 / 7.2 / 8.4 | 3.7 / 87.8 / 8.5 | 11.3 / 8.4 / 80.4 | 0↔2 · 41.4% |
| ret | BrainHero | ba_six | 85.4 / 6.1 / 8.5 | 3.4 / 87.2 / 9.3 | 11.2 / 7.2 / 81.6 | 0↔2 · 43.0% |
| ret | Graz | nine25 | 73.8 / 12.9 / 13.3 | 10.4 / 77.1 / 12.5 | 7.8 / 13.4 / 78.8 | 1↔2 · 36.8% |
| ret | Graz | ba_six | 76.2 / 12.2 / 11.7 | 10.1 / 76.9 / 13.0 | 9.2 / 12.8 / 78.0 | 1↔2 · 37.5% |

| Cohort | Stack | 0 lost pp | 1 lost pp | 2 lost pp |
| --- | --- | --- | --- | --- |
| new | nine25 | 3.61 | 4.36 | 4.90 |
| new | ba_six | 3.58 | 4.29 | 4.89 |
| ret | nine25 | 3.48 | 2.92 | 3.40 |
| ret | ba_six | 3.20 | 2.99 | 3.37 |

### Cell size

| Windows | New cells | Ret. cells | Nine25 lost pp | ba_six lost pp |
| --- | --- | --- | --- | --- |
| 90 | 1 | 1 | 0.47 | 0.42 |
| 91 | 1 | 0 | 0.35 | 0.35 |
| 92 | 1 | 1 | 2.28 | 2.26 |
| 93 | 7 | 1 | 5.01 | 4.99 |
| 94 | 2 | 0 | 0.75 | 0.75 |
| 96 | 15 | 4 | 13.19 | 12.95 |
| 103 | 1 | 1 | 0.62 | 0.60 |

Graz has exactly 32 windows of each class in every cell. BrainHero class prevalence ranges from 31.18% to 34.44%. Most of the deficit lies in cells with 96 windows because they account for 19 of the 36 cells.

## Existing causal evidence

Fable Round 3’s eight held-out orders measured the nine25 checkpoint oracle at +2.09 pp held-new and +1.04 pp returning, or +1.56 pp [1.00, 2.25] at 50/50. The held-new advantage fell from +8.50 pp for deliveries 1–10 to +0.99 pp for 41+. The frozen checkpoint used future windows and measures available adaptation information. Last Push’s causal alternatives already tested 40% Team B, early Rebalance and delayed SessionEM. The selected configurations gave −0.09 to +0.07 pp at 50/50 on ba_six, with all intervals spanning zero. [Fable Step 0](fable_round3.md#held-out-ceiling-and-team-b-diagnostics-on-orders-7130871315) · [Last Push](last_push_21nov.md#held-out-result-on-7180871815)

## Native stage interim · 15:15 CET

The first two fit orders (71800–71801) have exact captured labels for both stacks and all 36 cells. For ba_six, SessionEM adds +1.64 pp held-new and loses −1.39 pp returning. The returning loss is −2.38 pp on BrainHero. Count filtering loses −0.13/−0.38 pp held-new/returning, and MANAS adds +0.55/+1.09 pp. The perfect-identity intervention adds +2.01 pp on P04 and +0.01 pp on P10. These are two-order diagnostics. The count-filter observation preregistered one additional terminal test before any candidate fit or held-out scoring. [Ledger](../../docs/LEDGER.md#bottleneck-review--21-november-registered-before-analysis)

## Native stages, calibration and retrospective ceilings

All 96 Leonardo files matched frozen rows and labels on orders 71800–71803 and 71808–71811, with 27,336 predictions per stack. Capture manifest SHA-256  (sorted file SHA-256 lines). Native scores reproduced the full 25% replay. Rows show adjacent decisions in the running Solver. Position cells show weighted BA delta followed by counts of wrong→right / right→wrong flips. The raw row gives BA% before any listed change.

**Attribution.** ba_six’s hard and Team B mixers add +3.94 pp held-new and +4.77 pp returning before the tail. The tail through CSP adds +2.50/+4.21 pp, and MANAS adds +0.39/+1.05 pp. SessionEM helps held-new (+1.33 pp) and hurts returning (−1.11 pp), chiefly BrainHero (−2.52 pp). Its returning right→wrong flips outnumber repairs 231 to 164. The returning personal head and own-space stages add +2.76 and +1.46 pp. Team B loses 2.17 pp on returning deliveries 1–10 and adds 3.87 pp overall. Count filtering has a small negative adjacent effect on both cohorts. CSP repairs held-new errors and never changes a returning prediction in this bank.

**Calibration.** CSP log scores use softmax, and probability outputs use their native normalization. Expected calibration error (ECE) uses ten equal-width confidence bins. BA, ECE, negative log likelihood (NLL), Brier score and biases use the same cell/class weights. Both stacks are underconfident. ba_six’s confidence trails BA by 9.18 pp held-new and 11.06 pp returning. On Graz it overcalls imagery by 2.2 pp on held-new and undercalls imagery by 1.8 pp on returning. Word generation is undercalled by 2.0 pp on held-new Graz. The bias columns show mean predicted probability and decision rate minus true rate.

**Retrospective ceilings.** The fixed-pool search blends the native terminal posterior with one stage or member in 5% steps. The interface oracle picks the same pool separately per true interface. The single-member choice uses one member per cell across orders. Options that select using truth measure headroom. The exact-prevalence correction changes only the terminal Bayes decision for balanced accuracy. The recognition intervention supplies true returning identity and full ramp from delivery 1 to the live shadow gate, so upstream fine-tuning and subsequent state can respond. This intervention uses private evaluation metadata for diagnosis.

### nine25

| Stage | New BA% | New flips +/− | Returning BA% | Ret. flips +/− |
| --- | --- | --- | --- | --- |
| Raw mixer | 67.51 | — | 70.79 | — |
| Hard mix | 67.51 → 69.99 | 984/455 | 70.79 → 71.69 | 136/80 |
| Team B blend | 69.99 → 71.61 | 913/565 | 71.69 → 75.64 | 404/162 |
| Personal route | 71.61 → 71.61 | 0/0 | 75.64 → 78.08 | 322/174 |
| Rebalance | 71.61 → 72.31 | 1109/948 | 78.08 → 79.19 | 338/270 |
| SessionEM | 72.31 → 73.62 | 921/641 | 79.19 → 78.19 | 164/224 |
| TSEM | 73.62 → 73.95 | 542/468 | 78.19 → 78.67 | 194/165 |
| Own-space | 73.95 → 73.95 | 0/0 | 78.67 → 80.03 | 160/78 |
| Count filter | 73.95 → 73.89 | 22/36 | 80.03 → 79.99 | 9/12 |
| CSP | 73.89 → 74.01 | 317/291 | 79.99 → 79.99 | 0/0 |

| Cohort | Stage | All Δ | BrainHero | Graz | 1–10 | 11–20 | 21–40 | 41+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new | Raw mixer | 67.51% | 69.31% | 65.72% | 64.53% | 63.51% | 67.44% | 67.99% |
| ret | Raw mixer | 70.79% | 76.05% | 65.53% | 69.90% | 66.11% | 71.11% | 71.11% |
| new | Hard mix | +2.48 | +2.97 | +2.00 | +0.71 (19/11) | +0.80 (24/15) | +1.01 (66/43) | +2.91 (875/386) |
| ret | Hard mix | +0.90 | +1.71 | +0.10 | +0.64 (5/3) | +0.63 (8/6) | -0.18 (11/12) | +1.08 (112/59) |
| new | Team B blend | +1.61 | +1.89 | +1.33 | +0.87 (41/31) | +2.97 (64/30) | +3.53 (119/39) | +1.31 (689/465) |
| ret | Team B blend | +3.95 | +5.43 | +2.47 | -2.16 (8/15) | +3.79 (19/7) | +5.27 (47/13) | +4.19 (330/127) |
| new | Personal route | +0.00 | +0.00 | +0.00 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) |
| ret | Personal route | +2.44 | +1.75 | +3.13 | +0.00 (0/0) | -0.09 (14/14) | +3.01 (40/21) | +2.69 (268/139) |
| new | Rebalance | +0.70 | +0.01 | +1.40 | +0.15 (52/50) | +0.01 (81/80) | +0.32 (146/138) | +0.84 (830/680) |
| ret | Rebalance | +1.11 | -0.09 | +2.31 | +0.27 (16/15) | +5.04 (32/16) | +2.68 (49/32) | +0.70 (241/207) |
| new | SessionEM | +1.31 | +0.93 | +1.68 | +0.00 (0/0) | +0.00 (0/0) | +0.47 (67/56) | +1.59 (854/585) |
| ret | SessionEM | -1.00 | -2.49 | +0.49 | +0.00 (0/0) | +0.00 (0/0) | -1.59 (15/25) | -1.06 (149/199) |
| new | TSEM | +0.34 | -0.26 | +0.93 | +0.00 (0/0) | +0.19 (10/8) | -0.32 (54/61) | +0.46 (478/399) |
| ret | TSEM | +0.48 | -0.05 | +1.01 | +0.00 (0/0) | -1.26 (5/9) | -0.04 (28/28) | +0.70 (161/128) |
| new | Own-space | +0.00 | +0.00 | +0.00 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) |
| ret | Own-space | +1.36 | +2.11 | +0.62 | +0.00 (0/0) | +1.32 (9/5) | +1.41 (13/4) | +1.45 (138/69) |
| new | Count filter | -0.07 | -0.10 | -0.04 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | -0.08 (22/36) |
| ret | Count filter | -0.05 | -0.09 | +0.00 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | -0.06 (9/12) |
| new | CSP | +0.13 | +0.30 | -0.05 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | +0.16 (317/291) |
| ret | CSP | +0.00 | +0.00 | +0.00 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) |

**Calibration and class bias**

| Cohort | Interface | BA% | Confidence% | ECE pp | NLL | Brier | Probability bias 0/1/2 pp | Decision bias 0/1/2 pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new | BrainHero | 75.04 | 66.27 | 8.77 | 0.624 | 0.359 | -0.2/+0.2/+0.0 | +1.1/-1.0/-0.1 |
| new | Graz | 72.98 | 65.49 | 7.49 | 0.665 | 0.382 | +0.8/+0.2/-1.0 | +1.9/-0.4/-1.6 |
| new | Both | 74.01 | 65.88 | 8.13 | 0.645 | 0.370 | +0.3/+0.2/-0.5 | +1.5/-0.7/-0.8 |
| ret | BrainHero | 84.32 | 72.21 | 12.11 | 0.457 | 0.253 | -0.1/+1.2/-1.0 | -0.4/+1.6/-1.2 |
| ret | Graz | 75.65 | 68.23 | 7.42 | 0.620 | 0.348 | -2.8/+1.3/+1.5 | -3.2/+1.9/+1.3 |
| ret | Both | 79.99 | 70.22 | 9.77 | 0.539 | 0.300 | -1.5/+1.2/+0.2 | -1.8/+1.8/+0.0 |

**Recognition**

| Cohort | Person | Streams | Correct | False/wrong | First min/median/max |
| --- | --- | --- | --- | --- | --- |
| new | P10 | 24 | 0 | 1 | 40 / 40 / 40 |
| ret | P04 | 16 | 16 | 0 | 11 / 11 / 41 |
| ret | P10 | 16 | 16 | 0 | 11 / 11 / 19 |

**Retrospective diagnostics**

| Cohort | Best member | Second | Third |
| --- | --- | --- | --- |
| new | core1 65.52% | core2 63.88% | raw net 43 62.09% |
| ret | rest 68.69% | Team B member 5 67.56% | Team B member 6 67.37% |

| Oracle / diagnostic | New Δ pp | Ret. Δ pp | 50/50 Δ pp |
| --- | --- | --- | --- |
| Best single member chosen per cell | -3.69 | -7.54 | -5.61 |
| Best member or keep terminal, per cell | +1.03 | +0.00 | +0.51 |
| Any correct member or terminal, per window | +25.55 | +19.77 | +22.66 |
| True cell class prevalence correction | +0.04 [-0.06, +0.14] | +0.00 [-0.07, +0.07] | +0.02 [-0.04, +0.08] |
| True identity from delivery 1 (full ramp) | -0.00 [-0.01, +0.00] | +0.60 [-0.35, +1.60] | +0.30 [-0.18, +0.80] |
| Best grid class offsets per cell, hindsight | +1.15 | +0.76 | +0.95 |

| Terminal pool | 50/50 Δ pp | Selected blend |
| --- | --- | --- |
| Global fixed hindsight pool | +0.51 | 20% Team B member 2 |
| Interface-known hindsight pool | +0.66 | BrainHero 65% blend · Graz 15% Team B member 2 |
| Increment from interface knowledge | +0.15 | Same pool family |

Returning forced-recognition deltas are P04 +1.52 pp and P10 -0.32 pp.

### ba_six

| Stage | New BA% | New flips +/− | Returning BA% | Ret. flips +/− |
| --- | --- | --- | --- | --- |
| Raw mixer | 67.51 | — | 70.79 | — |
| Hard mix | 67.51 → 69.99 | 984/455 | 70.79 → 71.69 | 136/80 |
| Team B blend | 69.99 → 71.45 | 926/613 | 71.69 → 75.56 | 389/153 |
| Personal route | 71.45 → 71.45 | 0/0 | 75.56 → 78.32 | 327/158 |
| Rebalance | 71.45 → 72.20 | 1099/927 | 78.32 → 79.25 | 332/275 |
| SessionEM | 72.20 → 73.53 | 919/635 | 79.25 → 78.13 | 164/231 |
| TSEM | 73.53 → 73.78 | 548/492 | 78.13 → 78.47 | 186/166 |
| Own-space | 73.78 → 73.78 | 0/0 | 78.47 → 79.93 | 169/81 |
| Count filter | 73.78 → 73.72 | 24/37 | 79.93 → 79.77 | 4/14 |
| CSP | 73.72 → 73.95 | 336/287 | 79.77 → 79.77 | 0/0 |
| MANAS | 73.95 → 74.34 | 447/365 | 79.77 → 80.82 | 133/69 |

| Cohort | Stage | All Δ | BrainHero | Graz | 1–10 | 11–20 | 21–40 | 41+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new | Raw mixer | 67.51% | 69.31% | 65.72% | 64.53% | 63.51% | 67.44% | 67.99% |
| ret | Raw mixer | 70.79% | 76.05% | 65.53% | 69.90% | 66.11% | 71.11% | 71.11% |
| new | Hard mix | +2.48 | +2.97 | +2.00 | +0.71 (19/11) | +0.80 (24/15) | +1.01 (66/43) | +2.91 (875/386) |
| ret | Hard mix | +0.90 | +1.71 | +0.10 | +0.64 (5/3) | +0.63 (8/6) | -0.18 (11/12) | +1.08 (112/59) |
| new | Team B blend | +1.46 | +1.76 | +1.15 | +0.68 (42/34) | +2.44 (61/33) | +3.57 (124/43) | +1.16 (699/503) |
| ret | Team B blend | +3.87 | +5.59 | +2.15 | -2.17 (8/15) | +4.09 (20/7) | +6.09 (52/13) | +3.96 (309/118) |
| new | Personal route | +0.00 | +0.00 | +0.00 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) |
| ret | Personal route | +2.76 | +2.30 | +3.22 | +0.00 (0/0) | +1.19 (18/14) | +3.30 (40/19) | +2.98 (269/125) |
| new | Rebalance | +0.75 | +0.08 | +1.42 | -0.03 (53/53) | +0.45 (83/77) | +0.00 (137/136) | +0.92 (826/661) |
| ret | Rebalance | +0.93 | -0.88 | +2.73 | -0.92 (14/17) | +2.79 (28/19) | +1.87 (47/35) | +0.80 (243/204) |
| new | SessionEM | +1.33 | +0.94 | +1.71 | +0.00 (0/0) | +0.00 (0/0) | +0.42 (65/55) | +1.62 (854/580) |
| ret | SessionEM | -1.11 | -2.52 | +0.29 | +0.00 (0/0) | +0.00 (0/0) | -1.77 (15/26) | -1.18 (149/205) |
| new | TSEM | +0.25 | -0.32 | +0.83 | +0.00 (0/0) | +0.01 (9/9) | -0.58 (55/68) | +0.40 (484/415) |
| ret | TSEM | +0.33 | -0.21 | +0.88 | +0.00 (0/0) | -0.92 (8/11) | -0.33 (28/30) | +0.53 (150/125) |
| new | Own-space | +0.00 | +0.00 | +0.00 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) |
| ret | Own-space | +1.46 | +2.41 | +0.52 | +0.00 (0/0) | +1.96 (10/4) | +1.40 (13/4) | +1.54 (146/73) |
| new | Count filter | -0.06 | -0.10 | -0.02 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | -0.08 (24/37) |
| ret | Count filter | -0.16 | -0.19 | -0.13 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | -0.20 (4/14) |
| new | CSP | +0.23 | +0.48 | -0.02 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | +0.29 (336/287) |
| ret | CSP | +0.00 | +0.00 | +0.00 | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) | +0.00 (0/0) |
| new | MANAS | +0.39 | +0.30 | +0.47 | +0.47 (46/41) | +0.84 (47/38) | +1.19 (61/34) | +0.24 (293/252) |
| ret | MANAS | +1.05 | +0.63 | +1.46 | +2.81 (17/8) | +2.15 (11/4) | +0.80 (15/10) | +0.89 (90/47) |

**Calibration and class bias**

| Cohort | Interface | BA% | Confidence% | ECE pp | NLL | Brier | Probability bias 0/1/2 pp | Decision bias 0/1/2 pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new | BrainHero | 75.40 | 65.47 | 9.94 | 0.626 | 0.358 | -0.6/+0.3/+0.2 | +1.0/-1.2/+0.2 |
| new | Graz | 73.27 | 64.85 | 8.42 | 0.664 | 0.381 | +0.9/+0.3/-1.1 | +2.2/-0.2/-2.0 |
| new | Both | 74.34 | 65.16 | 9.18 | 0.645 | 0.370 | +0.1/+0.3/-0.4 | +1.6/-0.7/-0.9 |
| ret | BrainHero | 84.88 | 71.85 | 13.04 | 0.456 | 0.250 | -0.2/+0.8/-0.6 | -0.2/+0.8/-0.6 |
| ret | Graz | 76.76 | 67.67 | 9.28 | 0.605 | 0.341 | -2.7/+1.3/+1.4 | -1.8/+1.0/+0.7 |
| ret | Both | 80.82 | 69.76 | 11.07 | 0.530 | 0.295 | -1.4/+1.1/+0.4 | -1.0/+0.9/+0.1 |

**Recognition**

| Cohort | Person | Streams | Correct | False/wrong | First min/median/max |
| --- | --- | --- | --- | --- | --- |
| new | P10 | 24 | 0 | 1 | 40 / 40 / 40 |
| ret | P04 | 16 | 16 | 0 | 11 / 11 / 41 |
| ret | P10 | 16 | 16 | 0 | 11 / 11 / 19 |

**Retrospective diagnostics**

| Cohort | Best member | Second | Third |
| --- | --- | --- | --- |
| new | core1 65.52% | core2 63.88% | raw net 43 62.09% |
| ret | MANAS 69.80% | rest 68.69% | Team B member 4 67.56% |

| Oracle / diagnostic | New Δ pp | Ret. Δ pp | 50/50 Δ pp |
| --- | --- | --- | --- |
| Best single member chosen per cell | -4.27 | -7.51 | -5.89 |
| Best member or keep terminal, per cell | +0.84 | +0.00 | +0.42 |
| Any correct member or terminal, per window | +25.19 | +18.95 | +22.07 |
| True cell class prevalence correction | -0.08 [-0.19, +0.03] | -0.03 [-0.13, +0.07] | -0.05 [-0.13, +0.02] |
| True identity from delivery 1 (full ramp) | -0.02 [-0.05, +0.00] | +0.54 [-0.19, +1.34] | +0.26 [-0.10, +0.66] |
| Best grid class offsets per cell, hindsight | +1.11 | +1.02 | +1.06 |

| Terminal pool | 50/50 Δ pp | Selected blend |
| --- | --- | --- |
| Global fixed hindsight pool | +0.46 | 15% Team B member 2 |
| Interface-known hindsight pool | +0.54 | BrainHero 60% blend · Graz 5% MANAS |
| Increment from interface knowledge | +0.08 | Same pool family |

Returning forced-recognition deltas are P04 +1.23 pp and P10 -0.16 pp.

## Terminal tests on held-out orders

Each one-factor terminal rule used the preregistered 71800–71803 fit and 71808–71811 held split. The native class, CSP, count and MANAS controls reproduced the saved native labels and scores. The paired 80% intervals resample people and then orders. Each test preserves the preceding state and uses scores already produced for the current delivery. No terminal arm met the +0.15 pp and positive-lower-bound rule. The best ba_six point was the count bypass, +0.12 pp [−0.03, +0.28] at 50/50 and +0.27 pp returning. The class offsets and increased MANAS dose lost points.

| Stack | Terminal test | Frozen setting | Fit J Δ | New Δ [80%] | Ret. Δ [80%] | 50/50 Δ [80%] | Changed new/ret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nine25 | Class 1/2 log odds | -0.10/-0.10 | +0.03 | -0.16 [-0.41, +0.05] | -0.03 [-0.45, +0.38] | -0.10 [-0.31, +0.11] | 195/51 |
| nine25 | CSP evidence multiplier | 0.5 | +0.02 | +0.16 [-0.13, +0.40] | +0.00 [+0.00, +0.00] | +0.08 [-0.06, +0.20] | 168/0 |
| nine25 | Count-filter bypass | 1 | +0.06 | -0.07 [-0.21, +0.05] | -0.03 [-0.13, +0.07] | -0.05 [-0.11, +0.00] | 40/11 |
| ba_six | Class 1/2 log odds | -0.20/-0.10 | +0.22 | -0.13 [-0.51, +0.27] | -0.21 [-0.96, +0.52] | -0.17 [-0.59, +0.24] | 420/113 |
| ba_six | CSP evidence multiplier | 0.5 | +0.08 | -0.04 [-0.25, +0.15] | +0.00 [+0.00, +0.00] | -0.02 [-0.12, +0.08] | 157/0 |
| ba_six | Count-filter bypass | 1 | +0.12 | -0.03 [-0.15, +0.08] | +0.27 [+0.00, +0.53] | +0.12 [-0.03, +0.28] | 36/14 |
| ba_six | MANAS weight | 0.15 | +0.12 | -0.25 [-0.54, +0.05] | +0.13 [-0.34, +0.51] | -0.06 [-0.36, +0.21] | 284/62 |

Freeze receipt SHA-256 `a59a2a679f5e44f4ac3cd43a13520510705708e5158c77d430f06077bf15910f`.

## Actionable levers

1. **Gate SessionEM after recognition.** ba_six loses 1.11 pp returning at this step and gains 1.33 pp held-new. The current gate gives a feasible zero-marginal-inference way to target the returning loss. A 1 pp gain across returning cells would add 0.50 pp at 50/50. This change requires a live causal tail replay. The eight-fit/eight-held replay is running on Leonardo after four exact smokes. [Preregistration](../../docs/LEDGER.md#recognition-gated-sessionem--21-november-registered-before-running)
2. **Improve early P04 personal alignment or recognition ramp.** Immediate true identity buys +1.23 pp on P04 and −0.16 pp on P10 for ba_six, or +0.26 pp [−0.10, +0.66] at 50/50. All 32 returning streams eventually recognise correctly. The opportunity is timely and reliable personal evidence. This needs a causal EEG-derived change and independent returning people.
3. **Investigate count-filter latching under shuffled delivery.** Full terminal bypass is cheap and was tested above. Its ba_six gain is +0.12 pp with an interval spanning zero. Use the observed changed deliveries to test a more selective latch against fresh people before changing the current stage.

The strongest residual decoding target is the 41+ word generation confusion in held-new and persistent P04 Graz errors in returning. The fixed terminal pool gains +0.46 pp in hindsight on ba_six, and perfect interface knowledge adds only +0.08 pp within that pool family. The Scherer FP16 comparison gives −0.41 pp returning [−1.09, +0.22] and remains a transfer flag. [Scherer qualification](leonardo_fill_21nov.md#turing-only-manas-acceleration)
