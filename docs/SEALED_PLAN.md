# Sealed-phase plan and method description (draft, 20 October 2026)

Uploads are made by the user only. Recheck phase, quota and rules live in
Codabench before every upload.

## Facts that shape the plan

- Sealed phase: 28 Oct 2026 01:00 CET → 22 Nov 2026 01:00 CET (Codabench);
  1 submission per day, 10 in total, per account.
- Final ranking: "the best of each team's last five sealed submissions"
  (official `timeline.md`); leaderboard `Force_Best`.
- Sealed scores are **visible** to the team: the phase config hides the
  prediction files and the scoring-step output (logs), not the leaderboard
  score (`competition_bci.yaml`: "participants get their leaderboard score";
  duration column shown). A crash costs one of ten slots, with no log.
  (Corrected 25 Oct; earlier versions of this plan said scores were hidden.)
- Top three are audited by re-running training from the committed config
  (±2σ). Scherer 2015 (CC BY-NC-ND 4.0) for a cash prize is not answered
  by the organisers (issues #41, #52).

## Candidates (`runs/upload/`)

| ZIP | dev_loso | dev_session | all_loso | seen_session | External data |
|---|---:|---:|---:|---:|---|
| `sealed_candidate_ensemble_all_r6f_scherer.zip` (`accc74df…`) | 67.8 | 80.5 | 66.0 | 70.8 | Scherer 2015 |
| `sealed_candidate_ensemble_all_r6nf_noscherer.zip` (`1d347187…`) | 67.2 | 80.5 | 64.9 | 71.6 | none |

The two are within noise of each other (seen/unseen average 68.4 vs 68.3).
The no-Scherer package has no licence question, no external-data
declaration beyond "none" and a simpler audit (no external download).
**Recommendation: r6nf as the primary sealed submission**, unless the
organisers confirm before the deadline that CC BY-NC-ND data are admissible.

## Candidates after round 8 (25 Oct)

All pass the official replay in the worker replica (`scripts/cloud/worker_replay.sbatch`).

| Package | dl | ds | dl_all | all_loso (P04 / P09) | ss | replica s / 755 windows |
|---|---:|---:|---:|---:|---:|---:|
| r6nf (staged, primary) | 67.2 | 80.5 | 69.4 | 64.9 (48.5 / 69.6) | 71.6 | 66.7 |
| r8t4 (+4 learned-filter FBCNets) | 69.1 | 80.9 | 71.4 | 66.3 (47.8 / 70.7) | 70.9 | pending |
| r8t (+8 learned-filter variants) | 69.0 | 80.9 | 71.6 | 66.6 (48.7 / 71.2) | 70.5 | 107.4 |
| r6f (staged, Scherer data): **excluded** (user decision 25 Oct: no Scherer data) | 67.8 | 80.5 | – | 66.0 (– / –) | 70.8 | 66.8 |

## Upload sequence

Before 25 Oct 02:00 CEST (warm-up):
1. `warmup7_ensemble_official_r6f.zip`, expected 0.5243 (honest; checks the
   new fast code and the unaligned-FBCNet stream on the worker).
2. `sealed_candidate_ensemble_all_r6nf_noscherer.zip`, expected ≈ 0.9973
   (inflated: trained on the warm-up test participants). Purpose: the exact
   sealed bytes run on the worker; note the reported duration.
3. Optionally `sealed_candidate_ensemble_all_r6f_scherer.zip`, expected ≈ 0.9987.

Sealed phase (10 uploads, 1/day, 28 Oct – 21 Nov; best of the last five
counts, scores visible):
1. Uploads 1–5 measure distinct candidates on the sealed data itself (later
   sessions of the ten public participants + ten unseen), most robust first:
   r6nf (CPU, worker-run code), then r8t (learned-filter FBCNet family),
   then other candidates that pass the local rules. Each upload is a real
   readout of the ranking metric and of the worker duration.
2. Uploads 6–10 (the counted five) are the best-scoring packages, the best
   one at least once (inference is deterministic, so a resubmission
   reproduces its score). Never end with five untested packages.
3. Gaps of ≈ 1 point between sealed scores of similar packages can be noise
   of a single draw; prefer the locally better-evidenced package then.

## Method description (draft for the fact sheet / workshop)

**Model.** A probability-averaging ensemble of 21 members, run on CPU in
plain numpy/torch inside `submission.py`:
- v2: seven-band filter-bank covariances (1–4 Hz to high beta) mapped
  to the tangent space at a causally updated session reference, plus a
  causally centred < 1 Hz waveform, logistic regression (weight 0.2);
- 5 ShallowConvNets on Euclidean-aligned input, 5 unaligned
  ShallowConvNets, 5 aligned FBCNets and 5 unaligned FBCNets (0.04 each).

**Test-time adaptation (label-free, causal, reset per session).** Each
window is whitened by the running mean covariance of the session so far,
blended with the training mean as five pseudo-windows. Ensemble
probabilities are divided by the mean predicted class distribution of the
previous 30 windows (blended with a uniform prior of weight five).

**Training data.** PROTEUS (NEMAR nm000290 v1.0.0, CC BY 4.0), all ten
public participants, official preprocessing. Scherer package only: BNCI
2015-004 (Scherer et al. 2015, PLoS ONE 10:e0123727; licensor TU Graz;
CC BY-NC-ND 4.0), 2,130 windows preprocessed like PROTEUS, channels
interpolated to the PROTEUS montage, loss weight 0.1, used for the
unaligned ShallowConvNets and aligned FBCNets.

**Selection.** Leave-one-participant-out over the eight non-warm-up-test
participants and a later-session probe (P10 sessions 2–3); evaluation on
leave-one-participant-out over all ten and later sessions of P04/P10.

**Compute.** Final weights: about 20 nets × 1–3 min on one A100 or Apple
M4 Max (MPS), v2 under a minute on CPU (estimate: < 1 GPU-hour). Whole
development including sweeps: estimate on the order of 50–100 A100
GPU-hours on the Tübingen ML Cloud (to be summed from `sacct`). Inference:
CPU only, about 35 ms per window on an M4 Max and 62–64 ms on an
A100-node CPU (EPYC 7302).

**Reproduction.** `python scripts/build_sealed.py scripts/sealed_r6nf.json
OUT` (Galvani: `scripts/cloud/build_sealed.sbatch`) rebuilds r6nf from
scratch with the versions pinned in `requirements/cloud.txt` (neuroai
0.4.0, exca 0.5.29, braindecode 1.8.1, torch 2.14.1) and checks it against
the harness and the staged package. Rebuilds on MPS and an A100 agree with
the staged package on 2,654 of 2,655 windows
(`evidence/sealed-rebuild-2026-10-20`).
