# Regional mu and beta burst persistence (§407, 407b)

Bipolar band-envelope excursions improved the standalone decoder on the four previously exposed held people, but the six source folds did not reproduce the effect. Holding motor mass fixed hurt the frozen stack. Updating motor mass alone gave a small positive point estimate that both time-shuffled controls matched within uncertainty.

| Evaluation | Stock | Amplitude P | Duration X | Sample shuffle D | 100 ms block shuffle B |
|---|---:|---:|---:|---:|---:|
| Six source people, leave one out | — | 36.563 | 36.233 | 38.247 | 37.628 |
| Four held people, standalone | — | 33.738 | 34.980 | 33.940 | 33.751 |
| Four held people, full cognitive-only insertion | 81.216 | 80.719 | 80.721 | 80.759 | 80.692 |
| Four held people, full motor-only insertion | 81.216 | 81.144 | 81.368 | 81.352 | 81.420 |

Each value is cell-balanced accuracy (%) averaged across people. The full outcomes use eight pinned orders grouped by order family. P retains five amplitude statistics per bipolar derivation and frequency band. X, D and B additionally measure the longest high-amplitude run and the fraction of elevated samples belonging to runs of at least 100 ms.

| X versus | Source LOO | Held standalone | Full cognitive-only | Full motor-only |
|---|---:|---:|---:|---:|
| Stock | — | — | −0.495 [−0.840, −0.195] | +0.153 [−0.091, +0.394] |
| P | −0.330 [−1.221, +0.541] | +1.242 [+0.355, +2.170] | +0.002 [−0.153, +0.165] | +0.224 [−0.018, +0.456] |
| D | −2.014 [−2.987, −1.051] | +1.040 [+0.308, +1.772] | −0.038 [−0.297, +0.212] | +0.016 [−0.164, +0.195] |
| B | −1.395 [−2.874, +0.077] | +1.229 [+0.416, +2.043] | +0.029 [−0.112, +0.156] | −0.052 [−0.222, +0.124] |

Intervals are paired 80% resamples of people and order family members for the four held people, or of the six people for source LOO (10,000 draws, seed 407). Source LOO overlaps training sets. The standalone predictors are stateless. No returning-person or sealed evaluation.

Both full actors reproduced v3 replay OK and zero-change stock probabilities under independently checked orders. The separate motor and cognitive collectors agreed exactly on all standalone results and frozen stock metrics. The fit source manifest was ac37d612169655d9c518188640810edf52a75850da17b899747352092200cb1f, motor adba4a0b9f097ee846848517472d5d07438648a1d02aa4f0cd1860cec9ba321c, and corrected source LOO fc30ee0c3711e9507c26dc80ffb1f019ffb6d8b7bc775f66f0276a85e132607d. Both full actors used the same sealed member archive. Feature permutations use seed 407. Source revisions b3a59a9 and 9507c6f.

Slurm 3124204/05/07/08/09/10 completed the cognitive experiment. Source LOO scorer 3124206 lacked stream metadata and failed before producing a report. Corrected smoke and source jobs 3124232/33 completed. Motor smoke, orders and collector 3124229/30/31 completed. [Per-person results and prediction seals](burst407.json).
