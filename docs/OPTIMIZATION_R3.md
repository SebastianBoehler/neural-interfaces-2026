# Optimization round 3: leaderboard reading, seen participants, rebalancing

Run 18 October 2026 (Europe/Berlin). Protocols and metric as in
[OPTIMIZATION.md](OPTIMIZATION.md). Selection uses `dev_loso` and
`dev_session` only; `all_loso`, `seen_session` and `official` evaluate.

## What the warm-up leaderboard measures

Read 18 October, 08:29 CEST: 99 ranked entries. Ranks 1–24 score
0.75–1.00; one entry is named "full public data incl. P04/P09 – container
test". Training on the warm-up test participants is possible because the
warm-up data are public; our all-participant packages replay at 0.992–0.999.
Plausibly honest entries reach about 0.49–0.61. Our v1 (0.50) ranked 49th
overall, about 25th of the plausibly honest entries. P04 is 6 of the 8
warm-up cells, so these ranks say little about the sealed phase.

## Seen-participant protocol

The sealed phase scores later sessions of the 10 public participants as
well as 10 new ones. `seen_session` trains on all other participants plus
session 1 of P04 or P10 and tests sessions 2–3 (8 cells; evaluation only).

| Model | P04 s2–3 | P10 s2–3 | seen_session |
|---|---:|---:|---:|
| v1 | 49.8 | 55.1 | 52.4 |
| v2 | 48.9 | 66.4 | 57.7 |
| v2, C = 1 | 50.2 | 69.8 | 60.0 |

P04 gains little from its own first session, consistent with its slow
response being weak (round 2). Regularisation C = 0.03–3 moves
`dev_session` by ≤ 2.4 points (4 cells); C was left at 0.1.

## Causal class rebalancing

Every scoring cell has balanced classes, but a model transferred to a new
person often favours one class. At window t the ensemble's probabilities
are divided by m ** α, where m is the session's mean predicted probability
over windows 0..t−1 blended with a uniform prior weighted as k windows; then
renormalised. No labels, only earlier windows, reset per session
(`experiments/ensemble.py: rebalance`, `scripts/rebalance_probs.py`).

| Ensemble (v2 + aligned ×5 + raw ×5) | dev_loso | dev_session | all_loso | official |
|---|---:|---:|---:|---:|
| no rebalancing | 63.0 | 69.5 | 60.4 | 43.3 |
| α = 0.5, k = 2 | 64.3 | 75.8 | 62.1 | 49.0 |
| **α = 0.5, k = 5 (chosen)** | **64.3** | **75.8** | **62.0** | **49.1** |
| α = 0.5, k = 20 | 64.4 | 74.5 | 61.7 | 47.5 |
| α = 1, k = 5 | 63.3 | 77.5 | 62.2 | 48.9 |

α = 0.5, k = 5 was fixed on `dev_loso` (all α = 0.5 settings within 0.2)
with k matching the alignment prior. Rebalancing mostly corrects the nets
(`all_loso`: aligned ×5 +1.6, raw ×5 +1.8); v1/v2 alone change by
−0.3 to +0.6. Adding v1 to the ensemble or doubling v2's weight lowered
`dev_loso` and `all_loso` with or without rebalancing; the composition
is unchanged. Risk: if sealed sessions had strongly unbalanced classes,
rebalancing would cost accuracy; PROTEUS command runs are balanced.

## Packages

`scripts/add_rebalance.py` adds `rb_alpha`/`rb_k` to the round-2 packages
(nets unchanged) and checks every official-test window: the new
`submission.py` with rebalancing off reproduces the source package, and
with it on equals the harness `rebalance` of the member mix (both 100%).
Replays through the warm-up ingestion and scoring programs (Mac CPU):

| Package | Trained on | Replay cell mean | Ingestion |
|---|---|---:|---:|
| `ensemble_official_rb` | official train | 49.4883% | 27 s |
| `ensemble_all_rb` | all 10 participants | 99.8656% (leakage) | 25 s |

Submission files stayed byte-identical.

## Architecture sweep

Aligned inputs, A100, 40 epochs, seeds 0–2 (Galvani 2975397, 2975398).

| Network | dev_loso | dev_session |
|---|---:|---:|
| **FBCNet** | **59.2 ± 0.5** | 67.8 ± 2.9 |
| IFNet | 56.0 ± 1.9 | 69.3 ± 1.7 |
| TIDNet | 52.6 ± 0.5 | 61.6 ± 0.6 |
| EEGNeX | 50.9 ± 0.7 | 57.8 ± 2.1 |
| CTNet | 48.9 ± 0.2 | 54.0 ± 2.3 |
| EEGConformer | 45.9 ± 0.6 | 53.1 ± 0.4 |
| ATCNet | 45.2 ± 0.6 | 49.8 ± 1.6 |
| Deep4Net | 41.3 ± 0.8 | 38.2 ± 0.6 |
| ShallowConvNet (round 2, 5 seeds) | 53.7 ± 1.8 | 59.5 ± 1.7 |

FBCNet is the best single model on `dev_loso` (v2: 56.2). Offline mixes
of saved probabilities (seed-averaged members, equal weights, rebalancing
α = 0.5, k = 5):

| Mix | dev_loso | dev_session |
|---|---:|---:|
| v2 + aligned + raw (round-2 composition) | 64.3 | 75.8 |
| + FBCNet (chosen) | 65.1 | 77.5 |
| + FBCNet + IFNet | 65.3 | 77.5 |
| v2 + FBCNet + IFNet | 63.9 | 78.5 |

"+ FBCNet" was chosen on `dev_loso`; IFNet adds 0.2 and would need a
second plain-torch re-implementation. Evaluation on `all_loso` (one
FBCNet seed, run locally on MPS because the Galvani jobs 2975471/2975472
stayed queued behind the per-user job limit): FBCNet alone 55.4%; the
ensemble rises from 62.0 to **62.9%** with rebalancing (60.4 → 61.3
without) and improves 6 of 10 participants (P05 +8.4, P06 +5.3; P03,
P04, P09, P10 −0.5 to −2.0). The gain is modest but agrees with
`dev_loso` (+0.8), so `ensemble_all_r3` replaces `ensemble_all_rb` as
the sealed-phase candidate.

Also within noise and not adopted: rebalancing each member before mixing
(`dev_loso` 65.5), rebalancing both per member and after mixing (66.0),
and a geometric mean of members (65.5), against 65.1 for the chosen rule.

## Round-3 packages

`submissions/ensemble/submission.py` gained a plain-torch FBCNet (FIR
filter bank by FFT convolution, grouped spatial filters, batch norm, SiLU,
log-variance, linear head; effective max-norm weights exported). Against
braindecode it differs by ≤ 3.1e-6 in logits on random inputs with active
max-norm and BN statistics. `scripts/add_fbcnet.py` trains five FBCNets
(CPU), appends them to a rebalanced package (weights: v2 1/4, each net
1/20) and checks every official-test window against the harness (100%).

| Package | Trained on | Replay cell mean | Ingestion | CPU / window |
|---|---|---:|---:|---:|
| `ensemble_official_r3` | official train | 49.3996% | 35 s | 35 ms |
| `ensemble_all_r3` | all 10 participants | 99.8656% (leakage) | 30 s | 31 ms |

Staged ZIPs (`runs/upload/`): `warmup3_ensemble_official_rb.zip`,
`warmup4_ensemble_official_r3.zip`, `sealed_candidate_ensemble_all_rb.zip`,
`sealed_candidate_ensemble_all_r3.zip`. Nothing was uploaded by Claude.

## Training-time augmentation (not adopted)

`AlignedNet(aug=True)` applies a random circular time shift (≤ 0.25 s),
±10% per-window amplitude scaling and 10% channel dropout to training
batches only. FBCNet, MPS, seeds 0–2 (seed 0 also re-run without
augmentation on the same machine: 59.06%, identical to the A100 run):

| FBCNet | dev_loso | dev_session |
|---|---:|---:|
| plain (A100, 3 seeds) | 59.2 ± 0.5 | 67.8 ± 2.9 |
| augmented (MPS, 3 seeds) | 59.7 ± 0.5 | 73.2 ± 4.2 |
| ensemble with plain FBCNet | 65.2 | 77.5 |
| ensemble with augmented FBCNet | 65.0 | 77.3 |

The single-model gain is within noise and vanishes in the ensemble, so
the round-3 composition is unchanged. ShallowConvNet augmentation is
queued on Galvani (2975858) behind the per-user job limit.

## Current sealed-phase candidate

`ensemble_all_r3`: v2 (1/4) + five aligned and five raw ShallowConvNets +
five aligned FBCNets (1/20 each), causal rebalancing α = 0.5, k = 5,
trained on all ten public participants. Best estimates of its sealed
behaviour: `all_loso` 62.9% (unseen participants; one FBCNet seed) and
`dev_session` 77.5% (P10 later sessions, 4 cells). Baseline Torch-Linear
reference: 37.75% on the official split.
