# Release sensitivity to delivery and individual people

**Keep cap250 as the penultimate choice.** Public held-fold emulations favor cap250 over v5.1b even after simultaneously choosing the worst delivery scheme and removing the two most favorable people. On Scherer, the comparison of the exact archives changes sign when one person is removed. JS has no resolved advantage under the assumed private mix.

| Worst cap250 minus v5.1b, pp | All people | Remove one | Remove two |
|---|---:|---:|---:|
| Historical public new-person FULL-SYSTEM DEV, ten people | +0.878 | +0.576 (P09) | +0.349 (P04, P09) |
| Exact ZIPs on external Scherer, nine people | +0.040 | −0.346 (J) | −0.604 (C, J) |

Every minimum occurred under full-session shuffle. Public results use official cell weighting and enumerate 3, 30 and 135 scheme/subset combinations. External results enumerate 3, 27 and 108 combinations and retain the original equal-person metric. Each minimum averages orders within a scheme. These minima describe the recorded people. They are sensitivity bounds over the enumerated subsets, not confidence limits.

On public returning folds, cap250 minus v5.1b was +1.009, +3.093 and +1.323 pp on P04 under within/full/runs delivery. P10 gave +0.248, +1.386 and +0.412 pp. The public emulation passed both registered conditions. The Scherer extension met its predicted sign reversal. Its two orders use the same nine people and a single interface per day. [Public archive lineage and routing differences](lineage419.md) · [External montage, timing and schedule differences](scope416.md)

## JS release decision on held people

The assumed new-person share is 56.6%. Returning observations give P04/P10 weights of 20/80. Both cohorts use equal weights for within/full/runs delivery. This makes the fixed-JS comparison +0.041 pp, slightly above the lunch brief's estimate using the returning study's primary uniform weighting over eight orders.

For each of ten people, the release decision uses only the remaining people's B2_pool JS and cap250 scores. Leaving out P04 or P10 removes that person from both roles. The source estimate renormalizes each role separately, choosing JS when the mixed source difference exceeds zero. Held contributions keep their original target weights.

| Delivery | Fixed JS minus cap250, pp | Held decision minus cap250, pp | Folds choosing JS |
|---|---:|---:|---:|
| Equal schemes | +0.041 | +0.032 | 9/10 |
| Within each run | −0.073 | 0.000 | 0/10 |
| Full session | +0.174 | +0.174 | 10/10 |
| Run blocks | +0.021 | −0.138 | 6/10 |

Omitting P10 from source selection flips the equal-scheme decision to cap250 (source −0.161 pp). Omitting P08 leaves only +0.005 pp. The 80% people/order intervals span zero for fixed JS [−0.219,+0.208] and the held decision [−0.253,+0.130]. The latter satisfies the registered ≤+0.10 pp point prediction. Fixed arms and their underlying development pipelines were previously selected using these people. Only the final binary decision is cross-fitted. The same archived returning comparison and its gate limitations apply. [Returning bridge](js321_returning.md)

For each of 6,000 draws (seed 420), the scorer resamples shared orders within scheme and recomputes each source decision on the other recorded identities. It then resamples held people in strata of eight new-only and two dual-role identities. Dual-role identities retain linked new/returning contributions. This interval conditions source selection on the available people and quantifies variation in held evaluation and delivery.

## Reproduction

Public preregistration `9de6517`, external extension `520f770`. Stage [§419 inputs](lineage419.md) plus `runs/s281_v66_cp/JS_*_B2_pool.{json,npz}` from the same Galvani §318 snapshot into local `runs/release420/`. From the repository root:

```sh
.venv/bin/python replay/check.py > runs/lineage419/replay.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python studies/18nov/release420/score.py
```

Replay printed OK. The original §419 command reproduced the tracked JSON byte for byte after extracting its guarded matrices. The release scorer checked every §419 input digest, both cohorts' person and order effects, JS sidecar order/row provenance, all ten original §321 new-person JS effects and both returning weighting summaries. The external schedule means reproduced §416. All comparisons used existing outputs and local CPU arithmetic. [Results, source scores by held person and original JS archive hashes](release420_result.json)
