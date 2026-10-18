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
