# Handoff: Track 2 optimisation state (19 October 2026, evening)

Read this first, then `AGENTS.md`, `OFFICIAL_CONTRACT.md` and
`docs/OPTIMIZATION_R6.md`. Goal: maximise the **sealed-phase** score
(final ranking, prizes for the top three). The warm-up board is not a
goal: its top ranks (1.00) come from training on the public warm-up test
participants P04/P09.

## Key dates and quotas (recheck live in Codabench before any upload)

| Item | Value |
|---|---|
| Warm-up closes | 25 Oct 2026, 02:00 CEST |
| Sealed phase | 28 Oct 2026, 01:00 CET → 22 Nov 2026, 01:00 CET (website says 21 Nov AoE) |
| Warm-up quota | 5 / day, 100 total; 3 used (1000481, 1000482, 1000615) |
| Sealed quota | 1 / day, 10 total; 0 used |
| Evaluation limit | 60 min on one A100, inference only, `submission.py` + non-Python files |

Uploads are done by the user. Claude never uploads without explicit
authorisation.

## Challenge review (19 October, late; sources: Codabench API 17982, Terms, GitHub issues)

- Official repo: pinned `a66bc23` is still upstream `main` (no later
  commits). Worker image `neural-compet:v3` unchanged since 12 Oct.
- **Ranking rule:** the Timeline page says the final rank uses the best of
  each team's **last five** sealed submissions; the Codabench leaderboard
  is `Force_Best`. Plan for "last five": never spend a late slot on an
  experiment.
- **Sealed output is hidden** (`hide_output`, `hide_score_output`): a crash
  costs the day without a log. Only upload packages that already ran on the
  worker, i.e. verify the exact sealed package in warm-up before 25 Oct.
- Quotas are per Codabench account, not per team.
- **Audit (Rule 7):** top three must have their *training* re-run from the
  committed config and land within ±2σ of the sealed score. Our package
  is built by a chain of scripts (train_ensemble → add_rebalance →
  swap_external → add_fbcnet); this needs one reproducible pipeline,
  pinned versions (neuroai 0.4.0, exca 0.5.29, braindecode 1.8.1; see issue
  #61 for the exca 0.6 breakage) and a scripted Scherer download.
- Scherer 2015 (CC BY-NC-ND) for a cash prize: issues #41/#52 unanswered.
  Declare it either way; keep a no-Scherer fallback with the round-6 gain.
- Track 2 card: "Session 6 · unseen day"; the sealed sessions can lie
  further from session 1 than our `dev_session` probe (sessions 2–3).
- The FAQ explicitly allows gradient updates inside `predict` (test-time
  training), not only statistics-based alignment. `meta["device"]` is the
  A100; our package runs on CPU by design.

**Protocol caveat.** `all_loso` tests the 8 `dev_loso` participants plus
P04/P09, and `seen_session`'s P10 half is `dev_session`'s test data. After
six rounds of selection, evaluation is not independent of selection; only
P04/P09 in `all_loso` are untouched by selection. Treat ≤ 1.5-point gains as
unconfirmed and report P04/P09 separately.

## Task in one paragraph

Three mental commands (motor imagery, mental subtraction, word
generation) from 4 s, 41-channel, 120 Hz EEG windows (PROTEUS). Metric:
balanced accuracy per participant × session × interface cell, averaged
over cells. The sealed set mixes later sessions of the 10 public
participants with 10 unseen participants. `predict` gets one window at a
time per session, in order; label-free causal adaptation within a session
is allowed; state resets per session.

## Current best (sealed candidate)

`runs/upload/sealed_candidate_ensemble_all_r6_scherer.zip` (SHA-256
`cd6d2696…b4d4`), built from `runs/submissions/ensemble_all_r6/`:

- v2: 7-band session-aligned tangent space + causally centred 1 Hz slow
  waveform, logistic regression (`submissions/tangent`, `models.TangentSpace`).
- 5 aligned ShallowConvNets; 5 unaligned ShallowConvNets and 5 aligned
  FBCNets trained with Scherer 2015 (`~n~w0.1`); 5 unaligned FBCNets
  (PROTEUS only). Equal member-type weights (v2 0.2, each net 0.04).
- Causal windowed class rebalancing: last 30 windows, α = 1, k = 5.
- All plain numpy/torch in `submissions/ensemble/submission.py` (no
  braindecode at inference; ShallowConvNet and FBCNet re-implemented).
- 68 ms per window on Mac CPU → about 23 min for the estimated upper bound
  of 20,000 sealed windows. Keep new members inside this budget.

Scherer 2015 is CC BY-NC-ND 4.0; the user decided to use it (competition
not treated as commercial use). Declare it as external training data in
the method description. Fallback without it: `sealed_candidate_ensemble_all_r3w.zip`.

## Score history (cell-mean balanced accuracy, %)

| Candidate | dev_loso | dev_session | all_loso (unseen) | seen_session |
|---|---:|---:|---:|---:|
| Torch-Linear baseline (official split 37.75) | – | – | – | – |
| v2 alone | 56.2 | 68.3 | 54.9 | 57.7 |
| round 3 (`r3`) | 65.1 | 77.3 | 62.8 | 66.6 |
| round 5 + windowed (`r5w`) | 66.3 | 78.6 | 64.7 | 69.9 |
| **round 6 (`r6`)** | **67.8** | **80.5** | **66.0** | **70.8** |

Honest warm-up (official-train packages): v1 0.4973, round 3 0.4940
(worker-confirmed), round 6 0.5243 (local replay;
`runs/upload/warmup6_ensemble_official_r6.zip`, not uploaded).

## Protocols and rules of evidence (`experiments/protocol.py`)

- Select only on `dev_loso` (8 non-test participants) and `dev_session`
  (P10 sessions 2–3). Evaluate on `all_loso` (10 participants, unseen
  half) and `seen_session` (P04/P10 later sessions, seen half). Never tune
  on `official`/`all_loso`/`seen_session`.
- Noise: about ±1 point for ensemble comparisons. Prefer plateaus over grid
  winners; report the full neighbourhood (see `OPTIMIZATION_R4.md`).
- Every package must pass window-level agreement with the harness and an
  official ingestion + scoring replay (`scripts/replay_package.py`) with
  unchanged files before it is staged.

## How to run things

```bash
# local (bash, not zsh, for scripts/environment.sh)
.venv/bin/python experiments/run_cv.py v2 --protocols dev_loso,dev_session
.venv/bin/python scripts/add_rebalance.py SRC OUT 1.0 5 30
NI26_DEVICE=mps .venv/bin/python scripts/add_fbcnet.py SRC OUT --train all --raw --share 0.2 --rb-alpha 1 --rb-k 5 --rb-w 30
bash -c "source scripts/environment.sh && python scripts/replay_package.py PKG OUT cpu"
```

- Saved per-window probabilities: `runs/experiments/probs*`; offline mixes
  with `experiments/ensemble.rebalance(P, alpha, k, W)`. No tracked script
  reproduces the R4–R6 windowed mix tables yet (`external_probs.py` and
  `rebalance_probs.py` use the old unwindowed α = 0.5 rule); add one before
  the next selection round.
- Galvani ML Cloud: `ssh mlcloud-galvani`, workspace
  `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026`; sync with
  `rsync -az --exclude __pycache__ experiments scripts submissions …`.
  Jobs: `scripts/cloud/sweep.sbatch` (array, index = seed) and
  `scripts/cloud/external.sbatch` (one job, seeds in turn; supports
  `NAME@scherer2015~n~w0.1`). Results land in `runs/experiments/results-*.jsonl`
  and `probs*/`.
- Pitfalls: SSH needs the user's 2FA when the control master expires
  (open a terminal tab with `ssh -fN mlcloud-galvani`). The user's other
  project fills the per-user job limit, so jobs can wait hours; MPS on the
  Mac is a workable fallback for FBCNet (about 7 min per dev_loso seed).

## What did not work (do not repeat without a new angle)

Augmentation (single-model gains vanish in the ensemble), more seeds of an
existing member, weaker v2 regularisation and weight tuning (≤ 1 point),
unsupervised participant identification for personal models (cross-session
drift ≈ between-person differences), within-session self-training,
frozen REVE embeddings, Dreyer 2023 (MI-only sessions break alignment),
IFNet/TIDNet/EEGNeX/CTNet/Conformer/ATCNet/Deep4 aligned (weaker than
FBCNet). Details: `OPTIMIZATION_R3.md`–`R6.md`.

## Prioritised next steps

1. **Worker check of round 6.** Ask the user to upload
   `warmup6_ensemble_official_r6.zip` (expected 0.5243) before 25 Oct: first
   worker test of the unaligned-FBCNet stream and windowed rebalancing.
   Then upload the exact sealed package once to warm-up (trained on all
   public data, expected ≈ 0.9987, leakage) so that it has run on the worker
   before the hidden-output sealed phase.
2. **More diversity, the lever that just worked.** Unaligned variants of
   other architectures (`<arch>_raw` configs exist: tidnet, eegnex, ifnet,
   shallow, eegnet) on `dev_loso`/`dev_session`, 3–5 seeds on Galvani; add a
   member only if the windowed ensemble improves on both selection
   protocols. New architectures need a plain-torch re-implementation in
   `submissions/ensemble/submission.py` verified against braindecode
   (pattern: the FBC class, ≤ 3e-6 logit difference), or a check that the
   worker image's braindecode version matches 1.8.1.
3. **Scherer for the unaligned FBCNet** (61.5 single-model; was second-best
   as a fifth member) and member-level combinations, selected on dev only.
   `swap_external.py --kinds fbcnet_raw` now handles round-6 packages.
   Also build a no-Scherer round-6 fallback (r3w + unaligned FBCNet).
3b. **Unexplored families** (higher ceiling than more members): causal
   test-time adaptation of the nets (per-session BatchNorm statistics,
   AdaBN; entropy-minimisation updates, allowed by the FAQ), and GPU-side
   members (the A100 is idle at inference). Self-training with hard
   pseudo-labels failed, so prefer statistics-only updates first.
4. **Runtime**: measure the sealed candidate on an A100 node (CPU path
   dominates); keep the projected sealed time under about 30 min. The ten
   FBCNets recompute the same FIR filter bank per net (≈ 55% of window
   time); sharing it per input stream should save ≈ 40%. Also guard
   non-finite inputs (one NaN window poisons the rest of the session's
   state). Both change `submission.py` and need the full re-verification.
5. **Sealed-phase plan**: assume "best of the last five" counts: upload
   only worker-verified packages, keep the strongest for the final slots,
   and stop early rather than spend late slots on experiments. Write the
   method description (data, compute, licences) and a one-command training
   pipeline for the reproducibility audit.

## Repository state at handoff

- `main` at `539ed9e` (round-6 docs, pushed). Uncommitted after the
  review: this file, `swap_external.py` (windowed reference, aligned vs
  unaligned FBCNet kinds) and `add_fbcnet.py` (rebalancing settings now
  default to the source package's). Ask the user before committing.
- Legacy scripts that only document rejected experiments: `extract_reve.py`,
  `selftrain_probs.py`, `diagnose_slow.py`, `combine_probs.py`,
  `external_probs.py`, `rebalance_probs.py`; superseded:
  `cloud/export_ensemble.sbatch`, `train_tangent.py` + `submissions/tangent`
  (only for the old v1/v2 uploads). Docs link to them, so they are kept.
- Staged ZIPs in `runs/upload/` (ignored by Git), packages in
  `runs/submissions/`, data and caches in `data/` (ignored).
