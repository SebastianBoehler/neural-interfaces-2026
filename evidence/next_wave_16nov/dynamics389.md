# Whole EEG decoder with learned class-specific dynamics · §389

Causal session centering improved the whole-window decoder by **2.70 and 2.56 percentage points** on the two source splits. Correct history also beat a source-session donor. The adapted model scored **45.01% and 44.02%**, against **76.55%** for the complete JS decoder on the same ten people. The result supports session nuisance removal. It leaves the contribution of genuine temporal prediction unresolved.

| Whole-model accuracy / contrast (pp for differences) | Seed 389 | Seed 1389 |
| --- | ---: | ---: |
| Frozen / JS full systems | 76.28 / 76.55% | 76.28 / 76.55% |
| Shared dynamics C | 38.05% | 37.47% |
| Displaced predictive pairs P | 41.85% | 41.16% |
| Class-specific dynamics D | 42.31% | 41.46% |
| Strictly past session correction A | 45.01% | 44.02% |
| D − C | +4.26 [1.95,7.51] | +3.99 [2.09,6.03] |
| D − P | +0.46 [−1.69,3.13] | +0.30 [−2.37,2.70] |
| A − D | +2.70 [1.71,3.51] | +2.56 [1.64,3.68] |
| A − donor | +2.87 [1.95,3.58] | +1.85 [0.97,2.94] |
| Half-specific A | 45.77% | 45.85% |
| Half-specific A − whole-window A | +0.76 [−0.66,2.06] | +1.83 [0.88,2.92] |

Each class model predicted the next sample from 12 electrodes at four prior lags in three frequency bands. A separate source-only classifier read the per-electrode prediction error energies. The split kept template and classifier rows disjoint. Eight causal test orders covered all ten source-excluded people, including previously examined people. Time permutation preserved current EEG and the predictors’ marginal covariance. Its performance approached the fitted temporal model on both seeds, while the shared-template model lacked the class contrast features. The two-seed result therefore supports class-specific feature construction more clearly than predictive lag structure. The half-window follow-up was positive on both seeds; only seed 1389 had a positive lower interval.

All twenty whole and half person-fold jobs passed (3123664/3123665), along with input/actor smokes 3123660–63, original replay OK, and scorer 3123666. The frozen and JS scores reproduced their historical values, and the estimator reset to its initial state at every session. The 80% paired intervals resample people and orders. Source revision 992f121. [Mechanism](../../studies/18nov/dynamics389/fold.py) · [Source exclusion and hashes](dynamics389_before_labels.json) · [Complete estimates](dynamics389_result.json)
