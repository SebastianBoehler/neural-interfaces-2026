# §400 Calibration through the shuffled decoder

The learned session correction still helped at the first family mix. All three comparisons first turned negative when the decoder rebalanced that mixture against the preceding 30 windows. Each subsequent stage received its own arm's outputs. The reported deltas are percentage points of the official equal-cell accuracy, with equal weight on the three order schemes.

| B2_pool stage | Learned − native | Mean-only − native | Learned − source donor |
|---|---:|---:|---:|
| Individual model | +0.367 | +0.689 | +1.936 |
| Family mix | +0.283 | +0.189 | +0.454 |
| Causal rebalance | −0.233 | −0.224 | −0.136 |
| EM | −0.132 | −0.193 | −0.051 |
| TSEM | −0.250 | −0.196 | −0.042 |
| Count filter | −0.150 | −0.137 | −0.118 |
| Causal personalization | −0.277 | −0.257 | −0.238 |

The learned−native contrast fell **0.516 pp [80% −0.893, −0.068]** from mix to rebalance. Its decline at the initial mix was only 0.084 pp, refuting the preregistered mix-dominance prediction. Learned−donor lost 1.481 pp at the initial mix and a further 0.591 pp [−0.980, −0.171] at rebalance. Across six bases, the mix-to-rebalance changes were −0.402 [−0.689, −0.043] for learned−native and −0.570 [−0.846, −0.259] for learned−donor.

The weighted row transitions show how little of the individual result survives. Learned corrected a native error on 9.020% of row weight individually. Only 0.421 points of those 9.020 remained corrections at the family mix, and 0.116 points remained corrections at the complete endpoint. Of rows where both individual arms had equal correctness, the complete decoder introduced 0.668 points of injuries and 0.444 points of corrections. [All row transitions](stage400_lineage.json)

All 40 arm/order captures reproduced archived V6Fr decisions and probabilities within 2e-6 on all six bases. Capture jobs 3123969/70, replay 3123971, count jobs 3123974/75 and scorer 3123976 completed. Package CPU recomputation 3124043 matched archived stock, including CP scores within 2e-6. Atomic row audit 3124065 replaced 3124053 after a module-name collision. Every standalone and complete official score and nested interval matched the independent pinned §395c report. Study code checkpoint 78fe19d; attribution owner ni26_stage400_v1_20261118 with upstream ni26_policy395_chain_v1_20261118. All ten people had been examined previously; these are new-person fold replays of an archived stack. [Stage intervals and people](stage400_attribution.json)

## Crossed rebalancing state

§400b gives each learned or donor current mixture the native trajectory's class multiplier, then reruns EM, TSEM, the joint count filter and personalization. An independent reconstruction from the actual previous 30 native mixture rows (uniform prior count 5, exponent 1.15) matched every original native rebalanced probability to 1.89e−15 across 48 base/order cells. Each session starts with an empty history. Galvani snapshot ni26_stage400b_cross_v1_20261118. Smoke 3124061, 16 L/D cells 3124062, collector 3124063, state reconstruction 3124066.
