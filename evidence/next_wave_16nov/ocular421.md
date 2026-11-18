# §421 Recorded eye movements during word generation

Horizontal and vertical eye activity identified word-generation trials on 13 reserve participants with **58.59% accuracy** [80% people interval 53.85, 63.21]. Chance was 50%. After matching for eight eye-signal features, the same classifier achieved **58.97%** [55.90, 62.05]. Eye activity remains a plausible explanation for part of the external task response. This experiment cannot establish how much the packages relied on it.

The raw Shin2018 BrainVision recordings contain two electrooculogram (EOG) channels outside the model input. We aligned both at 1,000 Hz to each trial’s first EEG crop and checked all 1,560 trial markers against the exported labels and onsets. The export has 16 directly measured target electrodes and 25 interpolated electrodes. All package decisions came from the intact §415 A100 runs. The prediction digest and all 26 individual §415 differences reproduced. [Dataset and acquisition](https://www.nature.com/articles/sdata20183) · [EEG export and provenance](external_semantics351.md)

| External task selectivity in percentage points | Odd people (13) | Even reserve (13) |
|---|---:|---:|
| Intact cap250 minus v5.1b | +5.51 [2.95, 7.95] | +5.13 [1.67, 8.46] |
| Matched cap250 minus v5.1b | +7.69 [3.59, 11.79] | +2.82 [−1.54, 6.92] |
| Intact minus matched difference | −2.18 [−4.49, 0.13] | +2.31 [−0.77, 5.51] |
| Mean squared EOG pair distance remaining | 17.95% | 20.81% |

Matching selected five distinct trial pairs per run using EOG alone, before looking at either archive’s decisions. We retained both original delivery orders and the entire history that each intact worker saw. The intervals resample participants and their orders. All participants contributed to the previously published §415 aggregate. Groups are disjoint for the new EOG analysis. The EOG classifier learned its coefficients on the odd group only. Within-run feature standardization uses all 20 trials without labels for this retrospective instrument.

Both groups exceeded the prespecified 25% reduction in squared EOG distances. Matched effects were positive on average, but the even-group interval crosses zero. Attenuation changed sign between groups. The EOG classifier still detected the task after matching. Smaller pair distances therefore did not eliminate class information in the eye signals. These results establish neither a cortical origin nor ocular mediation of the package response. They provide no estimate of three-class competition accuracy.

Galvani smoke 3124679 and CPU score 3124720 completed in 77 and 79 seconds, respectively. Both replay checks returned OK. The analysis script SHA-256 starts with `054b0aff726d`; bootstrap seed 421. [Analysis](../../studies/18nov/ocular421/analysis.py) · [Results and per-person differences](ocular421_result.json) · [Frozen package comparison](selectivity415.md)
