# Optimization round 1: session-aligned decoders

Run 18 October 2026 (Europe/Berlin). Nothing was uploaded to Codabench.
Official submodule: `a66bc23d1a0e686b895c093a642b45c74970758b`, unchanged.

## Challenge bounds that shape the search

- **Sealed test = later sessions of the 10 public participants + 10 unseen
  participants.** No calibration data. Cross-subject and cross-session
  transfer both matter; warm-up (P04, P09) measures neither well.
- **Label-free causal adaptation within a session is allowed**: statistics of
  the current and earlier windows of the current session, reset per session.
  This permits causal Riemannian recentering / Euclidean alignment.
- Input is fixed: `(1, 41, 480)` at 120 Hz, microvolts, 4 s from cue+1 s.
  Only EEG; no participant, session or interface metadata reaches `predict`.
- Ranking: balanced accuracy per participant × session × interface cell,
  averaged equally. Multi-session participants weigh more (P04 = 6 of the
  8 warm-up cells; P10 = 6 of 20 dev cells).
- Compute: training uncapped; evaluation ≤ 60 min on one A100; single
  `submission.py` plus non-Python artifacts; nothing installed at evaluation.
- Published NeuralBench references on the cross-subject split are near
  chance: EEGNet 32.6%, CTNet 35.0%, fine-tuned REVE 36.0% (different
  metric aggregation; indicative only).

## Protocols (`experiments/protocol.py`)

All use the official metric helpers and stream test sessions in official
acquisition order; models must be causal within a session.

| Protocol | Train → test | Role |
|---|---|---|
| `dev_loso` | 7 of 8 non-test participants → held-out one | **model selection** |
| `dev_session` | dev minus P10 sessions 2–3 → those sessions | later-session probe |
| `official` | official train (6) → official test (P04, P09) | baseline comparison |
| `all_loso` | 9 of 10 → held-out one (28 cells) | evaluation of chosen candidates only |

Cell-level noise: one cell holds about 40 windows (≈7 points of balanced
accuracy standard error), so one-point differences are not meaningful.

## Results (cell-mean balanced accuracy, %)

| Candidate | dev_loso | dev_session | official | all_loso |
|---|---:|---:|---:|---:|
| Torch-Linear (official baseline) | – | – | 37.75 | – |
| Channel means + logistic regression | 44.5 | 45.5 | 39.6 | 44.3 |
| Filter-bank tangent space, no alignment | 42.2 | 38.7 | 40.1 | 42.7 |
| **v1**: 6-band tangent, session-aligned + centred means | 49.4 | 53.3 | **49.7** | 51.5 |
| 7-band (adds 1–4 Hz) + centred means | 51.2 | 59.6 | 48.2 | 52.6 |
| **v2**: 7-band + centred 1 Hz slow waveform | **56.2** | **68.3** | 45.7 | **54.9** |
| ShallowConvNet, aligned | 53.6 ± 2.0 | 58.3 ± 0.5 | – | 52.8 ± 0.9 |
| ShallowConvNet, raw | 53.2 ± 1.9 | 60.7 ± 0.7 | – | 52.3 ± 1.3 |
| EEGNet, aligned | 49.9 ± 0.2 | 56.9 ± 3.0 | – | 45.5 ± 0.7 |
| EEGNet, raw | 48.1 ± 1.4 | 55.7 ± 1.3 | – | – |

Linear models are deterministic. Deep models: A100, 40 epochs, AdamW,
one-cycle schedule; mean ± sample SD over seeds 0–2 (dev protocols) or
seeds 0–4 (`all_loso`). The official-split replay of Torch-Linear on the
A100 gave 38.12% (job 2975103), the Mac CPU runs 37.75% and 37.82%.

## Findings

1. **Causal session recentering is the main lever** (+6–7 points on dev,
   +8 on official). Training sessions are whitened by their own
   log-Euclidean mean; at test time the reference is the running mean of the
   session so far blended with the training mean as k = 5 pseudo-windows.
   A heavy prior (k = 20) loses 1–4 points; fast adaptation matters.
2. **Low frequencies carry most of the remaining signal.** The 1–4 Hz band
   and a session-centred < 1 Hz waveform add 7 points on dev. The slow signal
   is spread across the scalp: the two most frontal channels (AF3/AF4) give
   44.8% alone, all channels except them 49.5%, central/parietal 47.2%.
   It is therefore not purely ocular, but eye movements may contribute.
3. **v2 beats v1 on 9 of 10 participants in `all_loso`** (sign test
   p ≈ 0.02); P04 is the exception. Because P04 is 6 of 8 warm-up cells,
   v1 scores higher on the warm-up leaderboard (49.7 vs 45.7) while v2
   should be the better sealed-phase choice. Warm-up and sealed ranking can
   disagree; we optimise for sealed.
4. **Warm-up scores can be inflated by training on P04/P09.** v2 trained on
   all 10 participants replays at 99.2% on warm-up. Leaderboard entries far
   above about 55% probably include the test participants.
5. Tuning C (0.01–1), classifier (LDA vs logistic regression), training-time
   causal recentering and dispersion stretching changed results by ≤ 1.5
   points: within noise, so the simplest settings were kept.
6. Deep models trained from scratch are close to, but below, v2 and vary
   ±2 points between seeds. They are candidates for ensembling, not
   replacements.

## Submission validation

`submissions/tangent/submission.py` is a self-contained solver that stores
only numpy arrays (`weights.npz`; no pickled scikit-learn objects).
`scripts/train_tangent.py` fits through the harness, exports the folder and
asserts window-by-window agreement with the harness on the official test
streams. The exported folders were then replayed through the official
inference-only Benchopt run (`COMPET_SUBMISSION_DIR`, read-only files).

| Package | Trained on | Harness agreement | Official replay cell mean |
|---|---|---:|---:|
| v1 | official train | 100% | 49.7253% (identical to harness) |
| v2 | official train | 100% | 45.6604% (identical to harness) |
| v2 | all 10 participants | 100% | 99.1977% (leakage; not an estimate) |

The all-participant v2 ZIP holds `submission.py` and `weights.npz` at its
root (354,482 bytes uncompressed), and its full replay took about 10 s on
the M4 Max CPU. Submission files stayed byte-identical. Not yet replayed in
the official Docker image or on the Codabench worker.

Evidence: `evidence/optimization-2026-10-18/`.

## Next steps

1. Ensemble v2 with ShallowConvNet (probability averaging) under `dev_loso`.
2. Check the P04 failure: per-interface errors, slow-feature drift across
   sessions, k for the slow features separately from the covariances.
3. Use the resting-state runs for alignment priors (allowed by the FAQ).
4. Pretrained encoders (REVE probe, then fine-tuning) with alignment.
5. Decide the warm-up upload: an official-train model for an honest signal
   versus the all-participant model for the sealed phase. Upload only after
   explicit authorization and a live quota/phase check.

# Optimization round 2: probability ensemble and the P04 diagnosis

Run 18 October 2026 (Europe/Berlin). Nothing was uploaded to Codabench.
Next steps 1 and 2 of round 1 are done here. Protocols, metric and
causality contract are unchanged. Deep members ran on Galvani A100
(array jobs 2975257 and 2975282, seeds 0–4); linear models on the M4 Max.

## Ensemble

`experiments/ensemble.py` averages class probabilities of members fitted on
the same training windows; `TangentSpace` and `AlignedNet` gained
`predict_proba_stream` (logistic-regression probabilities, softmax of the
network logits). `run_cv.py` saves per-window probabilities, and
`scripts/combine_probs.py` scores weighted means of saved probabilities
(the same arithmetic as `Ensemble`). "×5" averages the five seeds of a
member before mixing it with v2.

| Candidate (selection) | dev_loso | dev_session |
|---|---:|---:|
| v2 | 56.2 | 68.3 |
| ShallowConvNet aligned / raw, 5 seeds | 53.7 ± 1.8 / 52.7 ± 1.9 | 59.5 ± 1.7 / 60.7 ± 1.4 |
| v2 + aligned net (one seed, `ens_ea`) | 59.8 ± 1.3 | 68.6 ± 1.0 |
| v2 + raw net (one seed, `ens_raw`) | 60.2 ± 0.8 | 70.6 ± 0.8 |
| v2 + aligned + raw (one seed each, `ens_both`) | 60.9 ± 1.5 | 68.5 ± 2.2 |
| v2 + aligned ×5 | 60.8 | 71.6 |
| v2 + raw ×5 | 60.9 | 70.1 |
| **v2 + aligned ×5 + raw ×5** (equal thirds) | **63.0** | 69.5 |

The last row was chosen on `dev_loso` (the main selection protocol; the
`dev_session` differences are inside its 4-cell noise) and fixed, and its
export jobs were submitted, before any ensemble was scored on `all_loso`.
With one ×5 member, v2 weights from 0.3 to 0.7 gave 58.5–62.0 on
`dev_loso` (best 0.4, at most 1.2 above 0.5); weights were not tuned. `combine_probs.py` also prints the
other combinations for `all_loso`/`official`; they were not used.

Evaluation of the chosen ensemble (cell mean, %):

| Model | all_loso | official (warm-up) | P04 all_loso | P09 all_loso |
|---|---:|---:|---:|---:|
| v1 | 51.5 | **49.7** | 45.4 | 54.8 |
| v2 | 54.9 | 45.7 | 42.5 | **62.5** |
| **Ensemble** | **60.4** | 43.3 | **46.2** | 60.4 |

On `all_loso` the ensemble beats v2 for 9 of 10 participants (sign test
p ≈ 0.02) and in 21 of 28 cells; the largest gains are P05 (+12.5), P02
(+13.0) and P10 (+9.3). On the official split (6 training participants)
it loses: the nets trained on 6 participants are weak on P09 (raw net
37.2 ± 1.4%, aligned 41.4 ± 1.0%), so P09 falls from 58.6 to 43.2 while
P04 rises from 41.4 to 43.3. The warm-up leaderboard would therefore rank
the ensemble below v1 and v2 even though it is the best sealed-phase
candidate on every protocol that trains on 7–9 participants. (Official
row: seed-0–4 sweep members; the separately trained exported package
scores 44.1.)

## Why v2 fails on P04 (diagnosis only)

`scripts/diagnose_slow.py` refits v2's classifier on the covariance block,
the slow-waveform block or both, and measures the slow features per
session. It reads P04/P09 labels to explain results; nothing was selected
with it.

- **The slow block is at chance on P04 everywhere.** `all_loso`: slow-only
  35.0% on P04 (32–39% in all six session × interface cells; chance 33.3%),
  covariance-only 45.6%, both 42.5%. For the other participants slow-only
  reaches 38–63% and adding it helps (overall 51.7 → 54.9).
- **It is a participant trait, not cross-session drift.** P04's
  class-contrast template of the session-centred slow features has a
  negative cosine with the mean template of the other participants in all
  three sessions (−0.31, −0.20, −0.16) but agrees with its own other
  sessions (+0.72 to +0.76). P02/P03 are at +0.72/+0.65. P10 is also near
  zero (−0.14 to +0.06; own sessions +0.68 to +0.75) and is v2's other weak
  participant.
- **P04's slow signal is weak even with its own labels**: within-session
  5-fold decoding of the slow block gives 55–57% (P02 77%, P03 82%, P10
  69–76%).
- **Causal centring is not the cause.** The lag of the causal session mean
  (k = 5) relative to the class contrast is 0.65–0.73 on P04, inside the
  range of other sessions (0.30–1.16). A separate prior for the slow
  features, k_slow ∈ {1, 20, 100}, changes `dev_loso` by ≤ 0.1 point and
  `dev_session` by ≤ 0.8; dropping the per-window DC offset (`v2_dc`)
  costs 0.7 points on `dev_loso`. Neither was adopted.

So the slow features encode a response whose topography or polarity
differs between people; when it does not match the population, it costs
about 3 points. The ensemble dilutes this (P04 42.5 → 46.2 in `all_loso`)
because the nets do not depend on that block. A label-free fix would need
an unsupervised mismatch signal; none is tested yet.

## Submission validation

`submissions/ensemble/submission.py` (≈210 lines, numpy + scipy + torch
only; ShallowConvNet re-implemented in plain torch with braindecode's
merged temporal-spatial kernel; no pickles) loads `weights.npz` with v2
(prefix `ts_`) and ten nets (`net0_`…`net9_`). `scripts/train_ensemble.py`
fits the harness `Ensemble`, exports the folder and asserts window-level
agreement with the harness on the official test streams (nets on CPU in
both). `scripts/cloud/export_ensemble.sbatch` runs it on Galvani, and
`scripts/replay_package.py` zips the folder (files at the root), extracts
it read-only and runs the warm-up phase's codabench ingestion and scoring
programs (the inference-only Benchopt run). The script reproduces the
round-1 v2 replay exactly (45.6604%).

| Package | Trained on | Harness agreement | Replay cell mean | ZIP bytes |
|---|---|---:|---:|---:|
| ensemble_official | official train | 100% | 44.1098% (identical to harness) | 2,999,935 |
| ensemble_all | all 10 participants | 100% | 99.8656% (leakage; not an estimate) | 3,001,000 |

Both packages: weights trained on an A100 (jobs 2975297/2975298), files
unchanged after replay, replay metrics identical on the Mac CPU and on a
native Linux A100 node (before the thread fix below).

**Evaluation time.** Benchopt calls `predict` once per window. On a
shared A100 node (load average 23–38) the first export ran at 0.11 s per
window (126 s for the 1,132 warm-up windows); a timing loop showed that
interleaved multi-threaded numpy/scipy and torch calls caused it (4
threads: 118–585 ms per window; 1 thread: 32 ms on the same node, 25–28 ms
on a Galvani CPU node). `load_model` now sets torch and threadpoolctl
(BLAS/OpenMP) to one thread; predictions are unchanged on all 1,132
windows for both packages, and the Mac replay takes 44–48 s. The sealed
set is at most about 106 sessions (20 participants × 6 sessions − 14
public) × ~190 windows ≈ 20,000 windows, i.e. about 10–11 min at 32 ms per
window, well under 60 min. The fixed `submission.py` was replayed on a
Linux Galvani CPU node (Xeon Gold 6226R, 4 CPUs): identical metrics,
41–56 s for 1,132 windows. Its replay on an A100 node is still open: the
interactive job (2975306) stayed queued behind the account's per-user job
limit and was cancelled.

The official-train ensemble later ran on the Codabench warm-up worker
(submission 1000481) with bit-identical metrics; see `docs/COMPETITION.md`.

Evidence: `evidence/optimization-2026-10-18-round2/`.

Round 3 continues in [OPTIMIZATION_R3.md](OPTIMIZATION_R3.md).

## Next steps

1. Decide the upload candidate. For the sealed phase the all-participant
   ensemble is the strongest; for an honest warm-up signal the
   official-train packages rank v1 > v2 > ensemble because of P04/P09.
   Upload only with explicit authorization and a live quota/phase check.
2. Run the package once in the official Docker image.
3. Unsupervised handling of the slow block (e.g. down-weight it when the
   session's slow features disagree with the population, without labels),
   selected on `dev_loso` only.
4. Resting-state alignment priors and pretrained encoders (round-1 items
   3–4).
