# Optimization round 6: regularisation, member weights, FBCNet diversity

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
four protocols (Galvani 2980792, 2980793), results below.

## FBCNet capacity and diversity (Galvani 2980792, 2980793)

Single models, seeds 0–4 (Scherer-trained aligned FBCNet: seeds 5–9):

| FBCNet | dev_loso | dev_session | all_loso | seen_session |
|---|---:|---:|---:|---:|
| aligned + Scherer (seeds 5–9) | 59.8 ± 1.0 | 70.3 ± 2.3 | 56.0 ± 0.7 | 56.0 ± 1.1 |
| unaligned | 60.4 ± 0.1 | 72.6 ± 1.2 | 56.0 ± 0.3 | 56.0 ± 0.8 |
| **unaligned + Scherer** | **61.5 ± 0.3** | 70.5 ± 0.8 | **57.1 ± 0.4** | 56.7 ± 2.4 |

Unaligned FBCNet with Scherer is the best single model so far. In the
windowed ensemble (seed-averaged members, equal member weights):

| Ensemble | dev_loso | dev_session | all_loso | seen_session |
|---|---:|---:|---:|---:|
| `r5w` (current) | 66.3 | 78.6 | 64.7 | 69.9 |
| aligned FBCNet with 10 seeds | 66.5 | 78.6 | 64.7 | 70.2 |
| aligned → unaligned FBCNet (+ Scherer) | 66.8 | 79.6 | 64.8 | 70.7 |
| + unaligned FBCNet + Scherer (5 members) | 67.6 | 79.2 | 65.8 | 70.7 |
| + unaligned FBCNet + Scherer, without raw ShallowConvNet | 66.3 | 80.0 | 64.7 | 71.0 |
| **+ unaligned FBCNet, PROTEUS only (5 members, chosen)** | **67.8** | **80.5** | **66.0** | **70.8** |

More aligned FBCNet seeds add nothing; the gain comes from a diverse fifth
member. The PROTEUS-only unaligned FBCNet was best on both selection
protocols and is chosen (+1.5 `dev_loso`, +1.9 `dev_session`); evaluation
agrees (+1.3 `all_loso`, +0.9 `seen_session`). It needs no external data.

## Round-6 packages

`scripts/add_fbcnet.py SRC OUT --raw --share 0.2 --rb-alpha 1 --rb-k 5
--rb-w 30` trains five unaligned FBCNets (seeds 0–4, Mac MPS) and appends
them with a 1/5 member share (v2 0.2, every net 0.04) to the windowed
packages. All 755 official-test windows agree with the harness (100%).
Official ingestion + scoring on Mac CPU, files unchanged:

| Package | Source | Trained on | Replay cell mean | Ingestion | CPU / window |
|---|---|---|---:|---:|---:|
| `ensemble_official_r6` | `ensemble_official_r3w` | official train | 52.4299% | 59 s | 55 ms |
| `ensemble_all_r6` | `ensemble_all_r5w` | all 10 participants (+ Scherer members) | 99.8656% (leakage) | 40 s | 68 ms |

`ensemble_official_r6` is our best honest warm-up candidate (previous
50.38%). `ensemble_all_r6` is the new sealed-phase candidate (Scherer 2015
declared as external training data). At 68 ms per window the estimated
upper bound of about 20,000 sealed windows needs about 23 min, within the
60-min limit. Staged: `runs/upload/sealed_candidate_ensemble_all_r6_scherer.zip`
(SHA-256 `cd6d2696…b4d4`) and `runs/upload/warmup6_ensemble_official_r6.zip`
(`558e7b50…2c8f`). Not uploaded.
