# Track 2 bottleneck review · 21 November 2026

**Interim, 14:29 CET.** Across 16 pinned orders, ba_six reaches **77.68%** at 50/50 against **77.32%** for nine25. P04’s four returning cells cost **7.11 pp** of ba_six’s **22.32 pp** error budget. Held-new word generation costs **4.89 pp**, the largest class loss in that cohort. Deliveries 41+ contribute **16.97 pp** of total loss.

## Scope and checks

The instrument uses real PROTEUS fold weights from `production-20261120-32a94ab7`. The 28 held-new cells span ten people. The eight returning cells span P04 and P10 on visits 2 and 3, with visit 1 enrolled. P04 and P10 also appear in the held-new instrument with different fit exclusions. Cell means weight true classes equally. Each held-new cell carries 1/56 of the joint score and each returning cell 1/16. The two-person returning instrument limits cohort generalization. [State](../../docs/STATE.md) · [Leakage audit](redteam_fable.md)

Both label banks from Leonardo `leonardo-fill-ba-six-21nov/grow/held` matched every label and delivery row in the independent `last-push-21nov/{fit,held}` zero arms and the regenerated full-session shuffle. The pinned v3 replay printed `OK`. Shuffled error analysis independently reproduces the +0.3566 pp comparison and its interval. The ba_six A100 fold path uses native BF16. The signed FP16-on-Turing archive follows that path on A100, with a separate 379-window bitwise A100 identity check and a 2,655-window zero-label-change Turing check on the all-data package. [Fold confirmation and precision checks](leonardo_fill_21nov.md)

The current fold tapes retain terminal decisions. Inference-only capture of intermediate scores is authorized on the same frozen fixtures and pinned delivery orders. Galvani smokes 3148143/3148144 check P01 on both stacks before the full capture. [Preregistration](../../docs/LEDGER.md#bottleneck-review--21-november-registered-before-analysis)

## Error budget

| Stack | Held-new | Returning | 50/50 BA | New lost pp | Ret. lost pp |
| --- | --- | --- | --- | --- | --- |
| nine25 | 74.26 | 80.38 | 77.32 | 12.87 | 9.81 |
| ba_six | 74.48 | 80.88 | 77.68 | 12.76 | 9.56 |

Paired ba_six gain in pp [80% interval], held-new / returning / 50/50: +0.21 [-0.04, +0.46] · +0.50 [+0.26, +0.74] · +0.36 [+0.17, +0.54].

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

Each cell shows window count · nine25/ba_six BA% · nine25/ba_six lost 50/50 pp.

| Cohort | Person | Visit | BrainHero | Graz |
| --- | --- | --- | --- | --- |
| new | P01 | 1 | 94 · 81.0/82.1 · 0.34/0.32 | 96 · 63.9/65.3 · 0.65/0.62 |
| new | P02 | 1 | 93 · 87.4/87.0 · 0.22/0.23 | 96 · 86.0/87.4 · 0.25/0.23 |
| new | P03 | 1 | 93 · 68.4/68.7 · 0.56/0.56 | 96 · 76.1/76.8 · 0.43/0.42 |
| new | P04 | 1 | 93 · 61.4/60.4 · 0.69/0.71 | 96 · 52.3/52.0 · 0.85/0.86 |
| new | P04 | 2 | 93 · 51.4/50.5 · 0.87/0.88 | 96 · 49.3/50.8 · 0.90/0.88 |
| new | P04 | 3 | 92 · 56.0/56.3 · 0.79/0.78 | 96 · 43.9/45.1 · 1.00/0.98 |
| new | P05 | 1 | 94 · 76.8/76.1 · 0.41/0.43 | 96 · 73.9/72.4 · 0.47/0.49 |
| new | P06 | 1 | 91 · 80.2/80.4 · 0.35/0.35 | 96 · 86.1/87.0 · 0.25/0.23 |
| new | P07 | 1 | 93 · 90.4/90.8 · 0.17/0.17 | 96 · 84.6/84.8 · 0.27/0.27 |
| new | P08 | 1 | 93 · 78.7/79.5 · 0.38/0.37 | 96 · 92.9/93.4 · 0.13/0.12 |
| new | P09 | 1 | 93 · 79.3/78.8 · 0.37/0.38 | 96 · 81.8/81.2 · 0.33/0.33 |
| new | P10 | 1 | 96 · 65.9/67.8 · 0.61/0.58 | 96 · 70.7/71.7 · 0.52/0.51 |
| new | P10 | 2 | 90 · 92.2/91.9 · 0.14/0.15 | 96 · 81.7/80.1 · 0.33/0.35 |
| new | P10 | 3 | 103 · 86.9/86.6 · 0.23/0.24 | 96 · 80.1/80.5 · 0.36/0.35 |
| ret | P04 | 2 | 93 · 72.1/72.9 · 1.74/1.69 | 96 · 66.3/67.5 · 2.11/2.03 |
| ret | P04 | 3 | 92 · 76.1/76.3 · 1.50/1.48 | 96 · 69.9/69.5 · 1.88/1.90 |
| ret | P10 | 2 | 90 · 94.7/95.6 · 0.33/0.28 | 96 · 81.4/82.2 · 1.16/1.11 |
| ret | P10 | 3 | 103 · 93.9/94.2 · 0.38/0.36 | 96 · 88.8/88.8 · 0.70/0.70 |

Largest ba_six cell losses in rank order: P04/2 Graz (ret, 2.03 pp), P04/3 Graz (ret, 1.90 pp), P04/2 BrainHero (ret, 1.69 pp), P04/3 BrainHero (ret, 1.48 pp), P10/2 Graz (ret, 1.11 pp).

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

## Existing causal evidence

Fable Round 3’s eight held-out orders measured the nine25 checkpoint oracle at +2.09 pp held-new and +1.04 pp returning (+1.57 pp at 50/50). The held-new advantage fell from +8.50 pp for deliveries 1–10 to +0.99 pp for 41+. The frozen checkpoint used future windows and measures available adaptation information. Last Push’s causal alternatives already tested 40% Team B, early Rebalance and delayed SessionEM. The selected configurations gave −0.09 to +0.07 pp at 50/50 on ba_six, with all intervals spanning zero. [Fable Step 0](fable_round3.md#held-out-ceiling-and-team-b-diagnostics-on-orders-7130871315) · [Last Push](last_push_21nov.md#held-out-result-on-7180871815)
