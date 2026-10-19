# Optimization round 6: regularisation, member weights, FBCNet variants

Started 19 October 2026 (Europe/Berlin). Reference: `ensemble_all_r5w`
(v2 + aligned ShallowConvNet ×5 + Scherer-trained raw ShallowConvNet ×5 +
Scherer-trained FBCNet ×5, equal member weights, windowed rebalancing
α = 1, k = 5, W = 30): `dev_loso` 66.3, `dev_session` 78.6, `all_loso`
64.7, `seen_session` 69.9. Selection on `dev_loso`/`dev_session` only.

## Weaker regularisation of v2

| v2 alone | dev_loso | dev_session | all_loso | seen_session |
|---|---:|---:|---:|---:|
| C = 0.1 (current) | 56.2 | 68.3 | 54.9 | 57.7 |
| C = 1 | 57.3 | 69.9 | 56.4 | 60.0 |
| C = 3 | 57.5 | 69.9 | 56.0 | 59.4 |

In the windowed ensemble, v2 → C = 1 gives 66.3 / 79.1 / 65.1 / 70.2 and
C = 3 66.3 / 79.9 / 64.9 / 70.1: ±0 on `dev_loso`, +0.2 to +0.5 elsewhere,
within noise. Not adopted on its own.

## Member weights (same saved probabilities)

| Weights (v2, aligned SCN, raw SCN, FBCNet) | dev_loso | dev_session | all_loso | seen_session |
|---|---:|---:|---:|---:|
| equal (current) | 66.3 | 78.6 | 64.7 | 69.9 |
| v2 0.15, others equal | 66.6 | 79.1 | 65.7 | 70.8 |
| v2 0.10, others equal | 66.3 | 78.1 | 65.8 | 70.7 |
| without v2 | 65.3 | 78.3 | 64.6 | 69.8 |
| **FBCNet ×2** (best on dev) | **67.0** | **79.6** | 65.3 | 69.8 |
| FBCNet ×1.5 | 66.6 | 78.6 | 64.9 | 69.9 |
| aligned SCN ×0.5 | 66.3 | 79.6 | 64.0 | 69.3 |
| raw SCN ×0.5 | 65.4 | 78.8 | 63.9 | 69.4 |
| v2 counted twice (v2 + v2 C = 1) | 63.7 | 77.8 | 61.9 | 67.8 |

Weight changes move the scores by about 0.5–1 point (noise level), except
over-weighting v2, which costs 2–3 points. FBCNet ×2 wins on the selection
protocols; lowering v2 to 0.15 evaluates better but was not the selection
winner. Rather than tune weights on noise, round 6 tests whether more
FBCNet capacity is a real gain: five more Scherer-trained FBCNet seeds
(5–9) and an unaligned FBCNet with and without Scherer, seeds 0–4, all
four protocols (Galvani 2980792, 2980793).
