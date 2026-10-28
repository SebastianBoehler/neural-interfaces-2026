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
