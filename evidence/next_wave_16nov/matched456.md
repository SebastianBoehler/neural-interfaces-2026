# §456 · Session memory and native readout on the three-seed cap250 reference

The learned kernel improved the historical cap250 full system on ten held-new public people. The two fits reached **76.3213% and 76.2622%**, versus **76.1039% stock**. For both seeds, genuine history beat stock and the same-age donor, with positive lower bounds on the registered paired 80% intervals. The covariance model and cap250's native readout changes had intervals spanning zero.

| FULL-SYSTEM DEV change on three-eye/three-rest stock | Seed 454, pp [80%] | Seed 1454, pp [80%] |
|---|---:|---:|
| Learned kernel − stock | +0.217 [+0.118, +0.310] | +0.158 [+0.049, +0.254] |
| Learned kernel − donor | +0.162 [+0.029, +0.281] | +0.205 [+0.029, +0.363] |
| Learned covariance − stock | +0.039 [−0.064, +0.142] | +0.012 [−0.124, +0.132] |
| Learned covariance − donor | +0.099 [−0.111, +0.303] | −0.014 [−0.170, +0.123] |

The kernel changed 294 and 231 of 21,240 stock decisions for seeds 454 and 1454. It improved 10/10 and 7/10 people, respectively. Held-new P04 and P10 gained +0.201 and +0.373 pp for seed 454 and +0.105 and +0.345 pp for seed 1454. [All per-order and per-person effects](matched456_learned.json)

| Cap250's native personalized readout | Matched difference, pp [80%] |
|---|---:|
| LDA class-prior correction − stock | −0.003 [−0.019, +0.018] |
| Reliability-weighted evidence − stock | +0.006 [−0.056, +0.075] |
| Reliability-weighted evidence − cyclic class control | +0.005 [−0.085, +0.107] |
| Both changes − class-prior correction | +0.024 [−0.036, +0.079] |

The prior correction changed 9 stock decisions and the reliability correction changed 75. All four native contrasts remain undecided. The corrections used cap250's actual pre-count V6Fr input, count-filtered probabilities and terminal personalized scores. [All native arm measurements](matched456_native.json)

An exploratory combination of the original §443 waveform actors showed why the baseline matters. Two-seed consensus scored 76.2040%, +0.10010 pp [−0.00078,+0.19386] versus stock, and +0.10575 [+0.01493,+0.18809] versus donor consensus. Removing the prior session mean scored 76.1062%, +0.00227 [−0.04537,+0.05350] versus stock. The prior-only control scored 76.2930%. Its comparison with stock was descriptive. [Consensus arms and all registered contrasts](matched456_consensus.json)

## Comparator, coverage and reproducibility

The comparator was §430's historical reconstruction with three eye seeds and three rest seeds. The auxiliary models trained on the other nine people in each fold. Each point uses 2,655 unique windows, 14 sessions and 28 equally weighted person/session/interface cells. The three delivery schemes receive equal weight. They include four within-run permutations, two whole-session permutations and two run permutations, with equal weight within each scheme. Each paired interval resamples people and then orders within scheme, weighting resampled people by their cell counts. The two fits share people and orders. No returning interpretation follows from P04/P10, because the fit excluded each person's entire public history. The actual staged ZIP uses separately trained weights. These public folds informed earlier decisions. [§430 lineage and limits](aux430.md) · [Earlier nine-eye reference](history443.md)

On Galvani, fresh recording-order replay printed OK on all six bases. Independent §443 reconstruction matched all 1,600 stored key arrays from the two seeds across both frozen and JS branches. The eight §430 archives matched the SHA-256 digests in the published report. The original §430 producer reproduced every archived terminal score bitwise while recording the exact arguments to its personalized stage. Each participating actor reproduced its historical decisions. The final scorer saved blind predictions and a hash receipt before reading held labels. First-order replay 3126856 and seven-order array 3126857 supplied the native stages. The first collector had a Python module-name collision. A second native smoke found shared process state in the original scorer's imports. Both failed before scoring. Native smoke 3126948 and collector 3126949 passed after isolating the unchanged native producer. The learned scoring jobs were 3126917/19/21/23 and the consensus collector was 3126906. [Restoration and scorer code](../../studies/18nov/matched456)
