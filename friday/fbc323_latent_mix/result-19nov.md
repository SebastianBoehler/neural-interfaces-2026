# F20 pooled waveform interpolation

Cross-person interpolation did not establish a gain over hard CE or same-session interpolation. Bounded cross-person accuracy was 58.7820% and 59.2085% held-new WHOLE DEV across seeds 0 and 1. Both registered contrasts remain undecided across seeds. [Preregistration](../../docs/LEDGER.md#f20--same-task-pooled-waveform-interpolation-across-source-people)

| Whole-decoder balanced accuracy | Seed 0 ordinary | Seed 0 bounded | Seed 1 ordinary | Seed 1 bounded |
|---|---:|---:|---:|---:|
| Exact hard CE | 58.5458% | 59.1613% | 58.3882% | 58.8654% |
| Within-session interpolation | 58.6325% | 58.6636% | 59.0834% | 59.5794% |
| Cross-person interpolation | 58.4198% | 58.7820% | 58.8193% | 59.2085% |

| Paired contrast in percentage points [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded cross-person minus hard CE | −0.3793 [−0.9213, +0.1280] | +0.3431 [−0.0495, +0.6892] |
| Bounded cross-person minus within-session | +0.1183 [−0.3873, +0.7946] | −0.3709 [−0.6972, −0.0535] |
| Bounded within-session minus hard CE | −0.4977 [−1.2494, +0.0881] | +0.7140 [+0.2773, +1.0881] |
| Ordinary cross-person minus hard CE | −0.1260 [−0.6606, +0.3813] | +0.4311 [+0.0148, +0.8093] |
| Ordinary cross-person minus within-session | −0.2127 [−0.7843, +0.6123] | −0.2641 [−0.6768, +0.1266] |
| Ordinary within-session minus hard CE | +0.0867 [−0.7786, +0.7215] | +0.6952 [+0.2328, +1.1105] |

Bounded cross-person minus hard CE was −0.0111/+0.0124 pp for P04 and −0.5524/+1.2884 pp for P10. Cross-person minus within-session was −1.1263/−0.9403 pp for P04 and −0.2304/+0.1687 pp for P10. The paired intervals resample people and delivery orders.

The cross-trained source cosine gap exceeded the same-session-trained gap by 0.000146 and 0.000111 across seeds. The predicted reduction was 0.015. This diagnostic uses genuine unmixed source windows and the first 64 registered source quartets. It did not show the predicted compression of same-task differences between source people.

Both active arms used the same 489,600 selected appearances from 1,958,400 actual source batch rows, exactly 25% in each fold and epoch. Selected rows served once as query and once as donor in each arm. The two maps had disjoint partner pairs. Summing per-fold distinct unordered pair counts gave 225,979 within-session and 242,998 cross-person pairs. Maximum true-row donor reuse was 23 in both arms. The strongest individual pair repeat count was five within-session and three cross-person. Actual head loss produced the required 3:1 anchor/donor gradient in each mixed arm.

Ferranti smoke 511294 passed in 12 seconds, and array 511295 completed all 60 fits. Galvani smoke 3132701 passed in 37 seconds. Array 3132702 sealed 20 complete held banks with 672 exact first-window checks, 1,344 restarts and maximum prefix posterior error 1.812e−6. All 84,960 hard-control decisions agreed with F16. Galvani 3132703 independently rebuilt all 20 same-task source tapes after the blind seal and completed the official whole-decoder score at 00:30:49 CET on 19 November 2026.

| Artifact in `ni26_friday_s1c_latent_mix_20261119` | SHA-256 |
|---|---|
| `actor.sha256` | `26206380249683ff1e32e458e67b64789f7376feda57381ec227197d473f3038` |
| `score.sha256` | `849c3dd7154160ec96b51290ffeaf3d146b2cec6643e60794de7b020f26543ef` |
| `before_labels.sha256` | `c90f1516e1e2fc7b24b03b4fc9ac559dcbba9e275ff33afea242618c6fcee543` |
| `whole_results.json` | `64678ecd1cb61b6869e4d54940b49581531e8dfe04162c661e4c687431094790` |
