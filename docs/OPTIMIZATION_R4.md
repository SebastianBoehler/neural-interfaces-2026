# Optimization round 4: rejected ideas and windowed rebalancing

Run 18–19 October 2026 (Europe/Berlin). Reference: the round-3 rebalanced
ensemble (`ensemble_all_r3`; v2 + aligned and raw ShallowConvNets ×5 +
FBCNet ×5), `dev_loso` 65.1, `dev_session` 77.3, `all_loso` 62.8.
Selection on `dev_loso`/`dev_session` only. Four ideas were rejected;
windowed rebalancing (last section) was adopted.

## Augmented aligned ShallowConvNet (Galvani 2975858)

Single models improve (`dev_loso`, 3 seeds): aligned ShallowConvNet 53.7 →
**57.0 ± 2.0** with augmentation; raw ShallowConvNet 52.7 → 50.8; FBCNet
59.2 → 59.7. In the ensemble every variant stays within ±0.5 of the
reference (swap aligned → augmented 65.5 / 77.8; add it as a fifth member
65.0 / 77.5; augmented ShallowConvNet and FBCNet 65.3 / 78.4). Not adopted.

## Personal models via unsupervised participant identification

Half of the sealed sessions come from the ten public participants. A v2
model trained only on the participant's own session 1 beats the global
model on their later sessions: P04 55.9 vs 48.9, P10 68.8 vs 66.4.
Using that requires recognising the participant without labels. A
7-band standardised log-covariance fingerprint identifies P10's later
sessions at every window count, but places P04's later sessions nearest
P09 until the whole session is seen, with margins ≤ 0.02. Distances of
seen sessions to their own session 1 (0.59–0.69) overlap the distances of
unseen participants to their nearest other participant (0.47–0.72), so
new participants cannot be rejected. Cross-session drift is as large as
between-person differences in this feature; routing was not built.

## Causal within-session self-training (`scripts/selftrain_probs.py`)

From window 30 on, a shrinkage LDA on 64 PCA components of v2's session
features is fitted to the pseudo-labels (rebalanced ensemble argmax) of
earlier windows and mixed with weight β:

| β | 0 | 0.2 | 0.35 | 0.5 |
|---|---:|---:|---:|---:|
| `dev_loso` | 65.1 | 65.4 | 61.9 | 60.0 |
| `dev_session` | 77.3 | 75.2 | 72.1 | 70.8 |

Pseudo-labels at 50–75% accuracy reinforce errors. Not adopted.

## Frozen REVE foundation model (`scripts/extract_reve.py`)

`brain-bzh/reve-base` (69 M parameters; loaded without gating), the
official probe's preprocessing, mean-pooled 512-d embeddings, logistic
regression: `dev_loso` 44.7 / `dev_session` 48.7 uncentred; causal session
centring and other C values were worse (36.5–43.6). As an ensemble member
at weight 0.1 / 0.2: `dev_loso` 64.6 / 64.1, `dev_session` 77.8 / 76.5.
Not adopted; the model licence was therefore not reviewed.

## Remaining lever

The ensemble saturates on the ten PROTEUS participants. The organisers'
Track 2 list adds public data: Scherer 2015 (BNCI2015-004; 9 participants,
30 channels, the same hand-MI, mental-subtraction and word-association
tasks; 19 channels shared with PROTEUS), Dreyer 2023 (87 participants,
27 channels, same lab and Graz protocol, left/right-hand MI only),
Zyma 2019 (mental arithmetic vs rest) and Stieger 2021 (MI, 399 GB).
This was done in round 5 ([OPTIMIZATION_R5.md](OPTIMIZATION_R5.md)):
Scherer 2015 helps slightly at low weight; Dreyer 2023 was rejected.

## Windowed rebalancing (adopted, 19 October)

PROTEUS runs are exactly class-balanced (10/10/10 in 30-trial runs,
12/12/12 in 36-trial runs); cue order is close to random (a class repeats
29% of the time vs 33% by chance). Run boundaries are not visible to the
model, so rebalancing now optionally uses only the last W windows
(`rebalance(P, alpha, k, W)`, package key `rb_w`). Rebalanced round-3
ensemble, saved probabilities, % cell mean:

| W / α | all / 0.5 (old) | all / 1 | 30 / 0.75 | **30 / 1** | 30 / 1.25 | 24 / 1 | 36 / 1 | 45 / 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_loso` | 65.1 | 65.5 | 65.4 | **66.1** | 64.9 | 66.2 | 65.6 | 65.8 |
| `dev_session` | 77.3 | 79.2 | 77.0 | **77.8** | 79.5 | 78.0 | 78.1 | 77.3 |
| `all_loso` | 62.8 | 63.7 | 63.8 | **64.2** | 63.9 | 64.6 | 63.9 | 64.0 |
| `seen_session` | 66.6 | 70.3 | 68.7 | **69.6** | 69.9 | 70.4 | 69.1 | 69.3 |
| `official` | 48.7 | 50.6 | 51.0 | **50.8** | 50.1 | 50.6 | 50.6 | 50.9 |

W = 30, α = 1, k = 5 was chosen on `dev_loso` among the first five
variants tried (W ∈ {all, 60, 30, 15}, α ∈ {0.5, 1}) and matches the run
length; the neighbourhood above was checked afterwards. Every α = 1 setting
with W 24–45 beats the old rule on all five protocols, so the gain is a
plateau, not one grid point; most of it comes from α = 1. Also tested and
rejected: within-run block structure (triples hold 1/2/3 distinct classes
62/573/248 times, close to random).

Gains stack with round 5 (Scherer members, `~n~w0.1`):

| Protocol | r3 old | r3 windowed | r5 old | **r5 windowed** |
|---|---:|---:|---:|---:|
| `dev_loso` | 65.1 | 66.1 | 65.6 | **66.3** |
| `dev_session` | 77.3 | 77.8 | 77.3 | **78.6** |
| `all_loso` | 62.8 | 64.2 | 63.5 | **64.7** |
| `seen_session` | 66.6 | 69.6 | 66.8 | **69.9** |

Packages (`scripts/add_rebalance.py SRC OUT 1.0 5 30`; nets and member
weights unchanged). Each passes two checks on all 755 official-test
windows: the new code with the source's own parameters reproduces the
source package (100%), and the new parameters equal the harness
`rebalance` of the member mix (100%). Official ingestion + scoring, Mac CPU,
files unchanged:

| Package | Replay cell mean | Ingestion | ZIP in `runs/upload/` |
|---|---:|---:|---|
| `ensemble_official_r3w` | 50.3800% | 67 s | `warmup5_ensemble_official_r3w.zip` |
| `ensemble_all_r3w` | 99.5968% (leakage) | 58 s | `sealed_candidate_ensemble_all_r3w.zip` |
| `ensemble_all_r5w` | 99.7312% (leakage) | 61 s | `sealed_candidate_ensemble_all_r5w_scherer.zip` |

`ensemble_all_r5w` is the new sealed-phase candidate (Scherer 2015 to be
declared as external training data); `ensemble_all_r3w` is the fallback
without CC BY-NC-ND data.
