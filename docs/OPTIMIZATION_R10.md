# Optimization round 10: multi-corpus pretraining for unseen participants

Started 28 October 2026. Reference: r9 (eight families, rebalanced),
selection 77.71, `all_loso` 66.95 (P04 49.6 / P09 69.7), `seen_session`
71.99, honest warm-up replay 0.536.

## Why

The warm-up board (docs/COMPETITION.md, 27 Oct) has 17 entries at 0.55–0.59
that could be honest. Our gap is on unseen participants: P04 alone is 6 of
the 8 warm-up cells. Tuning the current families on ten PROTEUS participants
is saturated (R8, R9: every probe within ±1). The untouched lever is the
training population. Earlier rounds only mixed external windows into the
3-class loss: Scherer 2015 gave +0.7 `all_loso` but is CC BY-NC-ND, and
Dreyer 2023 failed because its MI-only sessions break alignment. CBraMod,
LaBraM and REVE transferred poorly, because their inputs and pretraining
objectives differ from our families'.

## Method

Pretrain the trunk of our own families (ShallowConvNet, FBCNet, SincShallow,
FBLight) on many openly licensed motor-imagery corpora, then fine-tune on
PROTEUS with the unchanged recipe and harness:

- **Export** (`scripts/export_moabb.py`): the official preprocessing chain, the
  41 PROTEUS channels (spherical-spline interpolation), 120 Hz, 4 s
  windows covering each corpus's task interval, dataset-local labels.
- **Pretraining** (`experiments/pretrain.py`):
  - one head per corpus with a shared trunk; cross entropy is computed
    over the window's own head;
  - batches draw every corpus equally often;
  - alignment is per-session Euclidean alignment (EA) or per-corpus RMS (raw),
    as in AlignedNet.
- **Fine-tuning:** `run_cv.py NAME_ALIGN[_tf]_ptTAG[_lo|_lp]` loads the trunk
  and starts a fresh 3-class head. The tags select the recipe: lr 1e-3, lr
  3e-4 (`_lo`), or a 5-epoch linear probe then lr 3e-4 (`_lp`).
- **No leakage:** pretraining sees no PROTEUS data, so one checkpoint per seed
  serves every fold.
- **Packaging:** the architecture is unchanged, so `build_sealed.py` and the
  submission code need no changes.

## Corpora (licences checked 28 October; re-verify before shipping)

| Corpus | Participants | Classes | Licence |
|---|---:|---|---|
| PhysionetMI (Schalk 2004, EEGMMIDB) | 109 | L/R hand, both hands, feet, rest | ODC-By 1.0 (PhysioNet page) |
| Cho2017 (GigaDB 100295) | 52 | L/R hand | CC0 (GigaDB policy) |
| Lee2019_MI (OpenBMI, GigaDB 100542) | 54 × 2 sessions | L/R hand | CC0 (GigaDB policy) |
| Dreyer2023 (Zenodo 8089820) | 87 | L/R hand | CC BY 4.0 |
| Schirrmeister2017 (High-Gamma) | 14 | L/R hand, feet, rest | CC BY 4.0 (GIN LICENSE.txt) |
| Shin2018WG (DepositOnce 10.14279/depositonce-5830) | 26 | word generation vs baseline (two 4 s windows per 10 s trial) | CC0 (DepositOnce badge) |
| Shin2017A / Shin2017B (TU Berlin hBCI) | 29 | L/R hand MI / mental arithmetic vs rest (two windows per 10 s trial) | GNU GPL; MOABB `accept=True`, accepted by the user on 28 Oct |

Not used:

- **Scherer 2015:** CC BY-NC-ND (user decision).
- **Stieger2021:** 399 GB; its licence has not been checked.

## Results (seeds 0-2, wave `m7` = the seven corpora without Shin 2018)

Single families (rebalanced, seeds 0-2 averaged; selection = mean of
`dev_session` and `dev_loso_all`):

| Family | sel | ds | dl_all | dl | all_loso (P04 / P09) |
|---|---:|---:|---:|---:|---|
| shallow_ea | 63.78 | 65.77 | 61.80 | 57.74 | 58.76 (49.4 / 56.6) |
| shallow_ea_ptm7 | 68.43 | 74.54 | 62.32 | 61.55 | 58.13 (44.4 / 57.5) |
| shallow_ea_ptm7_lp | 68.39 | 73.45 | 63.32 | 62.34 | 58.87 (44.7 / 56.9) |
| shallow_raw | 63.80 | 67.45 | 60.14 | 56.00 | 56.26 (40.9 / 63.6) |
| shallow_raw_ptm7 | 67.97 | 72.93 | 63.01 | 62.73 | 57.21 (36.9 / 60.2) |
| fbcnet_ea | 71.89 | 78.59 | 65.18 | 64.23 | 61.00 (48.0 / 58.3) |
| fbcnet_ea_ptm7 | 73.41 | 80.73 | 66.10 | 64.52 | 61.16 (45.8 / 57.8) |
| fbcnet_raw | 70.48 | 77.14 | 63.83 | 63.27 | 60.48 (48.7 / 62.5) |
| fbcnet_raw_ptm7 | 71.10 | 78.20 | 64.00 | 63.86 | 59.85 (45.7 / 60.9) |

Pretraining helps every family most where PROTEUS training data are
scarce:

- `dev_loso` (7 training participants) gains up to +6.7.
- Later sessions of known participants (`dev_session`) gain up to +8.8.
- `dev_loso_all` (9 training participants) gains only +0.2 to +2.9.
- P04 loses 2-5 points in every family.

The r9 ensemble with pretrained members (seeds 0-2 throughout):

| Mix | sel | ds | dl_all | dl | all_loso (P04 / P09) |
|---|---:|---:|---:|---:|---|
| r9 | 77.55 | 83.22 | 71.88 | 69.71 | 66.98 (49.9 / 69.2) |
| ShallowConvNets → pt | 77.69 | 84.47 | 70.90 | 69.81 | 65.41 (46.2 / 68.1) |
| + aligned FBCNet → pt | 77.75 | 84.71 | 70.80 | 69.97 | 65.52 (47.1 / 68.1) |
| + unaligned FBCNet → pt | 77.72 | 84.44 | 71.01 | 69.91 | 65.63 (46.9 / 68.1) |
| r9 + pt ShallowConvNets (added) | 77.29 | 83.39 | 71.19 | 69.23 | 66.22 (48.5 / 69.7) |

Pretrained ShallowConvNets and FBCNets keep the later-session gain in the
ensemble (+1.2 ds) but lose about 1 point on unseen participants (dl_all,
all_loso, P04), so they are not adopted.

SincShallow is the exception: with only its family pretrained, every
protocol is up or flat (fresh seeds 3-5 below were not used for any
choice; learned-FIR FBCNets on seeds 2-3 there):

| Mix (r9 with ...) | seeds | sel | ds | dl_all | dl | all_loso (P04 / P09) | ss |
|---|---|---:|---:|---:|---:|---|---:|
| r9 | 0-2 | 77.55 | 83.22 | 71.88 | 69.71 | 66.98 (49.9 / 69.2) | 71.47 |
| SincShallow → ptm7 | 0-2 | 78.58 | 84.48 | 72.67 | 70.43 | 67.36 (49.0 / 69.2) | 72.15 |
| SincShallow → ptm7_lp | 0-2 | 78.25 | 83.71 | 72.79 | 70.37 | 67.54 (49.2 / 70.2) | 72.12 |
| SincShallow + FBLight → ptm7 | 0-2 | 78.25 | 83.96 | 72.55 | 70.12 | 67.05 (48.2 / 68.6) | 72.01 |
| r9 | 3-5 | 76.99 | 83.18 | 70.79 | 70.06 | 65.83 (48.2 / 69.2) | – |
| SincShallow → ptm7 | 3-5 | 77.85 | 84.20 | 71.50 | 70.03 | 66.40 (48.3 / 69.7) | – |
| SincShallow → ptm7_lp | 3-5 | 77.88 | 83.70 | 72.07 | 70.01 | 66.70 (48.3 / 68.1) | – |
| SincShallow + FBLight → ptm7 | 3-5 | 77.53 | 83.71 | 71.34 | 69.82 | 66.36 (48.8 / 69.1) | – |

Over six seeds, the SincShallow swap gains about +0.9 on selection, +0.5 on
all_loso and +0.7 on seen_session. This is small but replicated, so it is
R10 (`scripts/sealed_r10.json`, plain `ptm7` recipe).

**Other corpus sets (seeds 0-2):** mental-task corpora only (`mt`: Shin
2017B arithmetic, Shin 2018 word generation) and all eight corpora (`m9`)
do not beat `m7`:
- SincShallow selection: ptmt 69.09, ptm9 68.97, ptm9_lp 71.21 (vs ptm7_lp 72.29).
- r9 with SincShallow → ptmt: 77.47.
- r9 with SincShallow → ptm9_lp: 78.03.
- Adding fbcnet_ea_ptm9_lp to the m7 swap: 78.67 (+0.1, noise).
- P04 still loses 1-6 points in single families.

**Dev builds (28 Oct, uncommitted code `d89f133+r10`, Galvani 3016662/3016663).**
Both builds have harness agreement 1.0, and the official A100 replay leaves
their files unchanged.

| Build | Score | Replay |
|---|---|---|
| Sealed `sealed_r10_dev` | 0.9973 (inflated: trained on the warm-up test participants) | 70.7 s |
| Honest `warmup12_r10_tv_dev` (official train + validation) | **0.5319** (r9 0.5358) | 68.3 s |

The honest warm-up score is flat within noise: 8 cells, 6 of them P04, and P04
does not improve with pretraining. R10 therefore stays a day-4 candidate, not
a replacement for r9g.

## R11: domain-generalisation training (28 Oct, closed)

Two training options for unseen participants (`experiments/dg.py`; any
AlignedNet config name with the suffix `_swa` or `_cmix`):
- **Weight averaging (SWAD-style).** Average the weights of every step over
  the second half of training, then recompute batch-norm statistics.
- **Same-class mixing.** Blend each window with a random window of its class,
  mostly from another participant: lambda ~ Beta(0.4, 0.4), larger part kept,
  power preserved.

Galvani 3017434-3017441 ran all eight families at seeds 0-4
(`runs/experiments/probs_r11`).

**Swapping one family inside R10 (seeds 0-2).** R10 itself scores
78.58 / 72.67 dl_all / 67.36 all_loso. Every swap gives 78.33-78.76
selection, 72.20-72.83 dl_all and 67.14-67.58 all_loso, all within ±0.2.

**Applying one option to every family:**

| Option | sel | dl_all | all_loso |
|---|---:|---:|---:|
| `_swa` | 78.41 | 72.32 | 66.95 |
| `_cmix` | 78.14 | 71.82 | 66.52 |

**Single families alone (seeds 0-4).** Only ShallowConvNet-EA with `_cmix`
moves noticeably (selection 65.52 → 67.83, mostly `ds`); FBCNet-EA and the
pretrained SincShallowNet change by +0.4 at most. In the ensemble every
gain disappears. Closed.

Cross-session adaptation for unseen participants is excluded by the
official harness: every stream (one session) starts from a fresh copy of
the model, so no state carries over between sessions
(`benchmark_utils/streaming.py`).

**Slot plan.** Under Force_Best, seed copies of the best recipe add expected
score. R10 copies are therefore worth more than r9 copies.
`scripts/sealed_r10b.json` and `sealed_r10c.json` use r9b/r9c's seeds with
pretrained SincShallow trunks: pretraining seeds 5-14 is Galvani 3017635,
and the dev builds 3017646/3017647 depend on it.

**More seeds (Galvani 3017705, SincShallow ptm7 seeds 6-9).** The
pretrained family alone, over seeds 6-9:

| | sel | all_loso |
|---|---:|---:|
| Baseline SincShallow | 68.44 | 58.00 |
| Pretrained (ptm7) | 69.91 | 58.62 |

In the ensemble, seeds 6-7 (r9 → R10): selection 76.73 → 76.94, all_loso
65.86 → 65.86 (P04 46.9 → 47.1, P09 70.7 → 68.6).

R10's gain over r9 across the three seed sets:

| Seed set | sel | all_loso |
|---|---:|---:|
| 0-2 | +1.03 | +0.38 |
| 3-5 | +0.86 | +0.57 |
| 6-7 | +0.21 | 0.00 |
| **Mean** | **≈ +0.7** | **≈ +0.3** |

Positive in every set but small. Expected sealed effect: about +0.3 to +0.5.

## Bounds after R11 (28 Oct, night; R10 seeds 0-2 saved probabilities, no training)

| Bound (non-causal oracles use test labels) | all_loso | seen_session |
|---|---:|---:|
| R10 without rebalancing | 61.69 | 61.03 |
| R10 (W 30 rebalancing) | 67.36 | 72.15 |
| Exact per-run class counts, Hungarian (oracle run boundaries) | 67.74 | 73.96 |
| Best per-session logit bias (oracle) | 69.32 | 76.25 |
| Per-session affine logit map (oracle, in-sample) | 69.99 | 74.15 |
| Personal v2 at λ 0.3, oracle participant (r6nf mix) | – | 71.6 → 74.4 (P04 +4.0, P10 +1.6) |

- Output-level adaptation is spent: even with labels, fixing each session's
  bias or affine logit map gains ≤ 2.6 on unseen participants. The ≈ 15
  points to the within-session ceiling sit in the representation.
- Run structure is spent: every run is exactly balanced (10 or 12 per
  class), but exact-count assignment with oracle run boundaries adds +0.4.
- Training participants: `dev_loso` (7) → `all_loso` folds (9) gains +1.7
  on the same 8 test participants (5 of 8 up), ≈ +0.85 per in-distribution
  participant. Out-of-distribution data are worth far less (Scherer 2015:
  +0.7 for 9).
- Seen-participant routing stays blocked: session-mean covariance and log-PSD
  distances between sessions of the same participant (P04, P10) overlap
  completely with those between participants, at 20, 60 or all windows.
- Literature/dataset search (28 Oct): no public EEG corpus has motor
  imagery, subtraction and word generation in one session (only Rathee 2021
  MEG). Gradient TTA with entropy + marginal-diversity loss (T-TIME,
  arXiv 2412.07228) is the one literature-backed representation-level
  method not yet tried here; its reported +3 to +6 is over EEGNet+EA without
  rebalancing.

## R12: label-free test-time training (T-TIME-style, 28 Oct, closed)

`experiments/ttime_eval.py` (Galvani 3017870/3017871, shallow_ea and
fbcnet_ea, seeds 0-2, `runs/experiments/probs_ttime`). After predicting
window t, one Adam step on the last B windows of the session minimises
mean entropy minus the entropy of the mean prediction. Each session starts
from a fresh copy; every variant is paired with a frozen control `tt0`
that uses the same trained nets. `tt0` reproduces the earlier family scores
exactly.

| Variant (params, lr, B) | FBCNet-EA sel / dl_all / dl | ShallowConvNet-EA sel / dl_all / dl | R10 with both swapped: sel / dl_all / dl |
|---|---|---|---|
| tt0 control | 71.89 / 65.18 / 64.23 | 63.68 / 61.59 / 57.89 | 78.71 / 72.94 / 70.48 |
| n10 (BN affine, 1e-3, 8) | 69.84 / 63.24 / 63.78 | 65.31 / 61.56 / 57.00 | 77.43 / 72.17 / 70.96 |
| a1 (all, 1e-4, 8) | 68.33 / 60.02 / 61.01 | 60.67 / 54.13 / 53.07 | 77.75 / 70.93 / 69.51 |
| a3, a10, a3b16 | ≤ 68 / ≤ 58 / ≤ 57 | ≤ 53 | ≤ 77.6 |

All rebalanced (W 30), as in the packages. Without rebalancing, `n10` lifts
FBCNet `dev_loso` by about +2.5 on every seed (59.2 → 61.8). That is only
the class-bias correction rebalancing already gives (64.2). On top of
rebalancing, every adaptation strength loses, and stronger updates lose
more. Runtime was 2–7 ms per window on an A100, so cost was not the
problem. Closed: this matches the oracle bounds above.

## R13: Shin 2017 A+B in the 3-class loss; per-interface alignment (29 Oct, closed)

**Per-interface alignment (oracle).** v2 with session references, and
rebalancing, kept apart for Graz and BrainHero using the true interface of
each window (rebalanced; plain session alignment → per interface):
`dev_session` 69.50 → 67.14, `dev_loso_all` 58.17 → 58.26, `dev_loso` 57.22
→ 56.44. Even with the oracle there is no gain, so an interface detector is
not worth building.

**Shin 2017 A+B as PROTEUS-labelled data** (`scripts/merge_shin_ab.py`,
Galvani 3020050–3020055, seeds 0-2, `runs/experiments/probs_shin`). The same
29 participants did hand MI (A) and subtraction (B) in alternating sessions
of one day. Their windows were merged into one alignment session per
participant, labelled MI → 0 and subtraction → 1 (rest dropped), and added
to every training fold (`NAME@shin2017ab`, 5220 windows, about twice
PROTEUS).

| Family (rebalanced) | base sel / dl_all | + Shin sel / dl_all | R10 with it swapped: sel / ds / dl_all / dl |
|---|---|---|---|
| R10 | – | – | 78.58 / 84.48 / 72.67 / 70.43 |
| v2 | 63.84 / 58.17 | 49.37 / 48.08 (w 0.3: 50.82) | 78.12 / 85.05 / 71.19 / 71.48 |
| fbcnet_ea | 71.89 / 65.18 | 60.44 / 58.85 (w 0.3: 65.17; ft: 63.27) | best 78.68 / 85.26 / 72.10 / 69.96 (ft) |
| shallow_ea | 63.78 / 61.80 | 63.02 / 58.20 (w 0.3: 64.74 / 58.97) | best 78.97 / 85.51 / 72.43 / 71.55 (w 0.3) |
| fbcnet_raw / shallow_raw | 70.48 / 63.80 | 56.64 / 54.45 | 78.12 / 78.47 |
| sincshallow_ea / fblight_ea | 67.95 / 65.06 | 56.84 / 53.27 | 77.36 / 77.62 |

Every family loses on unseen participants (dl_all) when Shin is added. The
best R10 swaps gain only on `dev_session` (+0.4 sel, noise) and lose on
`dev_loso_all`. Likely cause: in Shin, class equals recording block (MI only
in sessions 0/2/4, subtraction only in 1/3/5). Any block-level drift (slow
offsets, impedance) predicts the class perfectly there, and the models learn
it. That hurts most for v2, which leans on the slow channel means. Aligning
each block separately would remove the confound, but also the task contrast
(the Dreyer 2023 failure). Closed: no public corpus gives a clean
within-session contrast between our classes.

**Per-session IAF band shift (closed, no run).** The alpha peak estimated
per session is unstable within participants (P10 7.5 / 12.5 / 8.0 Hz, P04
13 / 7 / 7 Hz), and many sessions have no clear peak, so bands cannot be
moved per session reliably.

## R14: rotation and frequency-scaling augmentation (29 Oct, open)

Training-time augmentation only (`experiments/dg.py`; run_cv suffixes
`_rot02/_rot05/_rot1`, `_fsc1/_fsc2`). After EA, the remaining between-
session difference is mostly a spatial rotation, so `rot` multiplies each
training window by a random orthogonal matrix exp(εA − εAᵀ), which keeps its
power. `fsc` stretches a batch in time by up to ±10/20%. Galvani
3023047–3023051, seeds 0-2, `runs/experiments/probs_r14` (rebalanced).

| Family | base sel / dl_all | best variant sel / dl_all | R10 with it swapped: sel / ds / dl_all / dl |
|---|---|---|---|
| fbcnet_ea | 71.89 / 65.18 | rot05 73.63 / 65.83 | 78.81 / 84.99 / 72.62 / 70.10 |
| shallow_ea | 63.78 / 61.80 | rot05 68.36 / 65.19 | 78.13 / 83.96 / 72.31 / 70.13 |
| sincshallow_ea_ptm7 | 69.85 / 62.00 | rot05 71.44 / 63.39 | 78.94 / 85.27 / 72.62 / 70.12 |
| fblight_ea | 65.06 / 58.97 | rot05 67.90 / 61.06 | 78.53 / 84.22 / 72.83 / 70.17 |
| fbcnet_raw / shallow_raw | 70.48 / 63.80 | fsc1 70.86 / 64.86 | 78.39 / 78.83 |

Rotation helps every aligned family alone (+1.6 to +4.6 sel); frequency
scaling and rotation of raw inputs do not. Swapped together into R10
(fbcnet_ea, sincshallow, fblight_ea → rot05): **79.31 / 85.78 / 72.84 /
70.04** against 78.58 / 84.48 / 72.67 / 70.43. With shallow_ea rot05 as well,
79.31 / 86.26 / 72.37 / 70.74. Adding the members instead of swapping them
loses (78.35–78.76). The three-member swap gains on every seed alone (swapped
members from one seed each: +1.11, +0.74, +0.59 sel; `dev_loso_all` +0.69,
+0.37, +0.38). `dev_loso` moves by −0.43, −0.48, +0.13. Candidate R14 =
R10 with these three rot05 swaps.

**Fresh seeds 3-5** (3023187; all deep members at seeds 3-5, TF members 3-4):
R10 77.69 / 84.72 / 70.66 / 70.13 → R14 **77.92 / 84.99 / 70.85 / 70.22**.
One seed at a time, sel gains +0.23, +0.52, +0.55 and `dev_loso_all` +0.21,
+0.52, +0.03. The gain is smaller than on seeds 0-2 but positive on all six
seeds; expect about +0.2 to +0.7.

**Dev build** (`scripts/sealed_r14.json`, Galvani 3023188, uncommitted tree
labelled `8780b12-dirty-r14`): harness agreement 1.0, official A100 replay
0.99601 with files unchanged, 68.8 s. `submission.py` is identical to R10
(`396e3866…`); only `weights.npz` changes. Staging needs a clean build from
a commit and a worker-replica run.

**Trainable-filter members** (3023191, seeds 0-1): fbcnet_ea_tf rot05 alone
65.23 → 69.61, fbcnet_raw_tf rot02 68.50 → 68.27. Swapped into R14 they
lose (79.12, 79.22, both 78.97 against 79.31), mostly on `dev_loso_all`.
These members stay unrotated: R14 keeps three swaps.

## R15: rotation in pretraining and in the tangent-space model (29 Oct)

**Tangent-space v2 with rotated copies (closed).** `TangentSpace(rot=ε)` adds
two copies of every training window with its tangent matrices rotated,
Q S Qᵀ in every band, and its slow waveform Q M (`models.rotate_features`;
it matches rotating the covariances to 1e-15). Alone: v2 63.84 → rot02
64.14, rot05 64.16, rot1 60.27 (sel; Galvani 3023391). Swapped into R14:
78.91 / 78.65 / 78.28 against 79.31. Closed.

**SincShallow trunk pretrained with rotation** (`pretrain.py --rot`, tag
`m7r` ε 0.05, `m7rr` ε 0.1; same seven corpora and recipe as m7; Galvani
3023383/3023384, fresh seeds 3023431). In place of R14's SincShallow
member (sel / ds / dl_all / dl):

| SincShallow member | seeds 0-2 | seeds 3-5 |
|---|---|---|
| R14: ptm7_rot05 | 79.31 / 85.78 / 72.84 / 70.04 | 77.92 / 84.99 / 70.85 / 70.22 |
| ptm7r_rot05 | **79.59 / 86.34 / 72.83 / 70.79** | **78.23 / 84.99 / 71.47 / 70.43** |
| ptm7r | 79.58 / 86.54 / 72.62 / 70.95 | – |
| ptm7rr_rot05 | 79.38 / 86.04 / 72.72 / 71.14 | – |
| ptm7rr | 79.12 / 85.78 / 72.46 / 70.94 | – |

One seed at a time (the swapped member only), sel changes by +0.72,
+0.59, −0.13, +0.42, +0.66, −0.63 and `dev_loso` by +0.31, +1.10, +0.22,
+0.51, +0.09, −0.52. Pooled over three seeds, the gain is +0.28 and +0.31
sel. Weaker evidence than R14 (four of six seeds), but positive in both
seed sets, and `dev_loso` improves with every rotation-pretrained trunk.
Candidate R15 = R14 with the m7r trunks.

## R16: rotation-pretrained FBCNet and FBLight trunks (30 Oct)

Same recipe as R15 (`pretrain.py --rot 0.05`, tag m7r; Galvani
3023731/3023732, seeds 0-2). In place of R15's member (sel / ds / dl_all / dl;
R15 79.59 / 86.34 / 72.83 / 70.79):

| Member | alone sel | in R15 |
|---|---:|---|
| fbcnet_ea_ptm7r_rot05 | 74.78 (rot05 73.63) | 79.80 / 86.56 / 73.04 / 70.99 |
| fbcnet_ea_ptm7r | 74.03 | 79.89 / 86.58 / 73.20 / 71.25 |
| fbcnet_ea_ptm7_rot05 | 74.36 | 79.83 / 86.56 / 73.10 / 70.89 |
| fblight_ea_ptm7r_rot05 | 67.82 (rot05 67.90) | 79.53 / 86.06 / 72.99 / 70.94 |
| fblight_ea_ptm7_rot05 | 68.00 | 79.29 / 85.82 / 72.77 / 70.90 |

FBLight: closed. FBCNet: +0.2–0.3, below the three-seed noise level
(docs/STRATEGY_R16.md §1). Seeds 3-5 are running (3023801) before any decision.

**R16 fresh seeds (3023801).** Pooled over seeds 0-5, R15 78.68 → 78.98
(ptm7r_rot05) and 79.01 (ptm7r); on seeds 3-5 alone 78.23 → 78.26 / 78.11.
At or below noise: not adopted.

## R17: rest-anchored unified pretraining (30 Oct, closed)

`pretrain.py --unified physionetmi,shin2017b,shin2018wg,schirrmeister2017`
(tag u9: m7 + Shin 2018WG; one shared MI/SUB/WG/rest block; its first three
outputs initialise the PROTEUS head; Galvani 3023804/3023805). Alone:
sincshallow ptu9_rot05 62.19 sel (seeds 0, 2) against 68.72 for ptm7r_rot05
on the same seeds; fbcnet ptu9_rot05 71.87 against 73.63. Swapped into R15:
77.83 / 78.04 / 78.92 against 79.59. The external task-vs-rest directions
do not transfer to PROTEUS's three-task contrast; the pretrained head even
hurts. Closed.

## R18: MixStyle and participant-adversarial training (30 Oct, closed)

`dg.MixStyle` (p 0.5, first two convolutions) and `dg.Adversary` (gradient
reversal on the final-layer input, λ 0.1 / 0.3); Galvani 3023864–3023869,
seeds 0-2. Swapped into R15 (79.59 / 86.34 / 72.83 / 70.79):

| Member | alone sel (base) | in R15 sel / dl_all |
|---|---|---|
| fbcnet_ea_rot05 ms5 / adv01 / adv03 | 72.84 / 71.04 / 67.81 (73.63) | 79.18 / 78.69 / 79.21 |
| shallow_ea ms5 / adv01 / adv03 | 63.97 / 60.09 / 55.12 (63.78) | 79.56 / 79.00 / 78.47 |
| sincshallow ptm7r_rot05 ms5 / adv01 | 69.12 / 70.40 (70.90) | 78.78 / 79.38 |
| fbcnet_raw ms5 / adv01 | 71.03 / 67.78 (70.48) | 79.41 / 78.92 |
| shallow_raw ms5 / adv01 | 63.83 / 61.58 (63.80) | 79.68 / 79.09 |
| fblight_ea_rot05 ms5 / adv01 | 67.00 / 68.05 (67.90) | 79.52 / 79.42 |

Participant-adversarial training hurts every family, more so with larger λ.
With only ten participants, removing participant information also removes
class information. MixStyle is neutral (±0.4, within noise). Closed.

## R19: per-session gaze templates (30 Oct, candidate)

Found by the ground-up review (docs/GROUND_UP_REVIEW.md §2.4, §5 rank 1)
and re-implemented independently in `experiments/session_em.py`.
- **View:** horizontal (F5+AF3+F3 − F6+AF4+F4) and vertical
  (AF3+AF4 − P1+P2+Pz) slow potentials in 16 bins per window, 32 values.
- **Template:** per session and causally, a soft EM estimates Gaussian
  class templates from the session's earlier windows. It is seeded by the
  rebalanced R15 posteriors, with class means shrunk to the session mean
  (m0 20) and covariance shrinkage 0.7.
- **Combination:** the template likelihood multiplies the posterior with
  exponent 0.5·n/(n+60), from window 10 on.

No training. Pre-registered config, R15 seed sets checked separately
(`scratchpad/verify_em.log`):

| | seeds 0-2 | seeds 3-5 |
|---|---|---|
| dev_loso_all | **+0.93** (shift-null max +0.57) | **+1.67** (null max +0.38) |
| nested LOPO selection over 12 configs, dla | +0.89 (se 1.26) | +0.95 (se 1.04) |
| dev_session | −0.89 (null mean −1.44) | +0.43 (null mean −0.62) |
| all_loso (diagnostic) | +0.77 | +1.45 |
| first 60 windows per session, dla | −0.41 | +0.84 |
| participants ≥ 0 (dla) | 4/8 | 6/8 |

The seen_session proxy changes by −0.18 (P10 +1.5, P04 −1.9).

**Reading.**
- The unseen-participant gain is real: it is above every circular-shift and
  noise null on both seed sets.
- The seen side is mixed. The stage itself costs −0.6 to −1.4 on P10 under
  the null, and the gaze signal wins only part of that back.
- Over the 18-config grid, mean(ds, dla) ranges −0.09 to +0.76 averaged over
  both seed sets. The frozen config gives +0.54.
- Sealed expectation ≈ 0.57 × (+0.9 to +1.7) + 0.43 × (−0.9 to +0.4)
  ≈ +0.3 to +0.8.

R19 = R15 + this stage, made by `scripts/add_session_em.py` (R15 weights
unchanged; new `submission.py` with the `em_*` keys).

**Band-limited EA reference (closed).** Nets whitened with the covariance
of their 4–40 Hz band-passed windows, input kept broadband (`ref_band`, suffix
`_rb440`; Galvani 3024080, seeds 0-2). Alone: fbcnet_ea_rot05 73.63 → 69.67,
fblight 67.90 → 64.66, shallow_ea 63.78 → 54.66 sel. In R15: 78.17 / 79.31 /
77.58 against 79.59. The broadband reference, dominated by the slow signal,
is what these nets rely on. The linear proxy in the review did not transfer.
Closed.

## Round 2 bottleneck review (30 Oct)

A second multi-agent pass looked at three things: literature on 2024–26
zero-calibration and recent EEG challenges, which closed levers rest on weak
evidence, and three CPU probes on saved six-seed R15 probabilities. Two
critics then tried to refute each candidate.

**Bottleneck.** The remaining loss is participant-specific class
directions. On the same rows, R15 trails the within-session ceiling by
13.3 on dla (MI 5.5, SUB 3.2, WG 1.9) and by 3.7 on ds. The biggest
single blocks:
- P10: MI recall 29 against 94;
- P03: all classes;
- P05 and P07: SUB.

**Ruled out in this round:**

| Lever | Result |
|---|---|
| 30–58 Hz power (subvocal EMG) | Alone at chance cross-subject (dla 33.6). Mixed in: inside the shift-null, mostly negative per participant |
| Per-participant member-family or input-subset routing | Positive in-sample; cross-fitted −0.2 to −2.6 dla |
| Amplitude as P04's explanation | P04's RMS (13.4–13.8 µV) equals that of P01/P02/P06, who score 70–82, and raw members are no worse on P04 than EA members. The "P04 19 µV" claim in GROUND_UP_REVIEW §2.3 is wrong |
| v2 slow-block scaling | Inside R15: ds −0.73, sel ≈ +0.06 |
| Population-template prior for the gaze EM | −0.6 sel on both seed halves |
| External same-task data (BCI III-V), imaginary coherency, labelled Procrustes oracle | Refuted by the critics: closed mechanism, dla −0.01 on one half, or no path to a sealed slot |

## R19e: gaze + ocular-event rates in the session EM (30 Oct, candidate)

The review's rank-1 joint view, which R19 had shipped only half of. It adds
16 causal, sign-free event rates to the 32 gaze values (48 in all):
- traces: 8 Hz low-passed first differences of horizontal
  (F5+AF3+F3 − F6+AF4+F4) and vertical (AF3+AF4 − C1+C2+Cz) frontal signals;
- events: |trace| above 4× and 8× the running per-window median, where the
  median is blended with a training prior g weighted as 5 windows;
- 4 time bins.

The EM config is unchanged (`experiments/session_em.py` `EventRates`).
These are saved-probability replays on R15, with the gates fixed before
the run (scratchpad `round2/verify/verify_r19e.py`):

| R19e minus R19 | seeds 0-2 | seeds 3-5 |
|---|---|---|
| dev_loso_all | **+1.26** (shift-null max +0.67, noise max +0.51) | **+0.53** (+0.22, +0.01) |
| dev_session | +0.84 | +1.83 |
| all_loso (diagnostic) | +0.80 | +0.50 |
| first 60 windows per session, dla | +0.64 | +0.35 |

- Pooled per participant on dla: +0.82 (se 0.61), with 7 of 8 participants
  ≥ 0. P08 is −2.95 and P06 +2.77.
- Nested leave-one-participant-out over the 12-config grid: +2.44 (se 0.66).
- g computed inside each fold gives the same result (+0.89).
- seen_session, R10-like proxy members, against R19: −0.11 (P10 +0.29, P04 −0.51).
- seen_session with R15's actual members (new six-seed files from Galvani
  3025249, `probs_r15ss`; diagnostic only): R15 71.10, R19 71.41, R19e
  72.35. Seeds 0-2: 70.84 / 71.82 / 72.48. Seeds 3-5: 71.24 / 71.81 / 71.96.
  P04 is 57.6 / 55.2 / 57.3, so the events give back what the gaze stage
  cost P04.
- Against R15: ds −0.05 / +2.27, dla +2.19 / +2.21, al +1.56 / +1.94.

All five gates pass. Caveats:
- The event definition had already been seen on these posteriors in the
  review, so this confirms a frozen definition rather than testing on
  fresh data.
- ds is a single participant.

Expected sealed gain over R19 is about +0.2 to +0.5.

`submission.py` computes the events in the stream; the `em_ev_*` keys
enable it. It matches `session_em` to 3e-16 on 755 real windows.
Config: `scripts/sealed_r19e.json`. Build:
`scripts/add_session_em.py --events` on the R15 clean package.
