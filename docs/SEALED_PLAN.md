# Sealed-phase plan and method description (draft, 20 October 2026)

Uploads are made by the user only. Recheck phase, quota and rules live in
Codabench before every upload.

## Facts that shape the plan (updated 26 Oct: upstream `2b53aec`, live Codabench)

- Codabench was down 20–26 Oct (PHP security issue, then a compute-side
  incident); the competition was migrated. Warm-up extended to **1 Nov 2026,
  23:59 CET** (5 per day, 100 total). No warm-up upload of ours happened yet.
- Sealed phase: **5 Nov 00:01 → 21 Nov 23:59 CET**; **at most five
  submissions per participant for this track, one per day**. "The best
  eligible score from you or your team is automatically displayed and
  retained for the final ranking" (`submission_rule: Force_Best`).
- Aggregate sealed scores stay visible; execution logs and prediction /
  scoring files are hidden. A crash costs one of five slots, with no log.
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
| r8t4 (+4 learned-filter FBCNets) | 69.1 | 80.9 | 71.4 | 66.3 (47.8 / 70.7) | 70.9 | 83.1 |
| r8t (+8 learned-filter variants) | 69.0 | 80.9 | 71.6 | 66.6 (48.7 / 71.2) | 70.5 | 107.4 |
| r6f (staged, Scherer data): **excluded** (user decision 25 Oct: no Scherer data) | 67.8 | 80.5 | – | 66.0 (– / –) | 70.8 | 66.8 |

Staged (25 Oct), each ZIP byte-identical to the one that passed the worker
replica: day 1 `sealed_candidate_ensemble_all_r6nf_noscherer.zip`
(`1d347187…`), day 2 `sealed_candidate_ensemble_all_r8t4.zip` (`f858e6c2…`,
built from `c625ecc` + `scripts/sealed_r8t4.json`, committed in `48f4419`;
Galvani 3002636/3002637). Projected sealed runtime at ≈ 21k windows: r6nf
≈ 25–30 min, r8t4 ≈ 30–40 min on the worker CPU (60-min limit).

**GPU builds (preferred, 25 Oct).** Same weights with the current
`submission.py`, whose FBCNets and ShallowConvNets also run on the harness
device (CUDA on the worker; CPU fallback when no GPU is passed). In the
worker replica (Galvani 3003404/3003405): labels identical to the CPU path on
all 2,655 windows (max |ΔP| 1.2e-7), 31 ms vs 83 ms per window (r8t4) and
25 vs 64 ms (r6nf); official replay 46.5 s / 40.6 s for 755 windows.
Staged: `sealed_candidate_ensemble_all_r6nfg.zip` (`9d430f0d…`) and
`sealed_candidate_ensemble_all_r8t4g.zip` (`725cfcdc…`); projected sealed
runtime ≈ 9–11 min. Plan: day 1 r6nfg (baseline score and the worker's real
duration), day 2 r8t4g; the CPU ZIPs stay as fallbacks.

## Upload sequence

Before 25 Oct 02:00 CEST (warm-up):
1. `warmup7_ensemble_official_r6f.zip`, expected 0.5243 (honest; checks the
   new fast code and the unaligned-FBCNet stream on the worker).
2. `sealed_candidate_ensemble_all_r6nf_noscherer.zip`, expected ≈ 0.9973
   (inflated: trained on the warm-up test participants). Purpose: the exact
   sealed bytes run on the worker; note the reported duration.
3. Optionally `sealed_candidate_ensemble_all_r6f_scherer.zip`, expected ≈ 0.9987.

Done 26 Oct: both sealed ZIPs ran on the real worker (1010506 r6nfg 0.9973 in 24.0 s, 1010508 r8t4g 1.0000 in 25.6 s; `docs/COMPETITION.md`).
Before 1 Nov (warm-up, 5/day): upload the exact sealed ZIPs to run them on
the real worker (GPU path, duration). Their warm-up scores are inflated
(trained on the warm-up test participants): r6nfg ≈ 0.997, r8t4g ≈ 1.000.

Sealed phase (five uploads, best one counts, scores visible): no
resubmissions are needed, so every slot is a distinct, worker-verified
strong candidate; a weaker one costs nothing. Order: r8t4g (best evidence),
r6nfg (baseline), then the best remaining distinct candidates, informed by
the scores so far.

## Method description (draft for the fact sheet / workshop; updated 25 Oct)

**Model (r8t4; r6nf = the same without the last family).** A
probability-averaging ensemble of 25 members in plain numpy/torch inside
`submission.py`; the nets run on the harness device (CUDA on the worker,
CPU otherwise):
- v2: seven-band filter-bank covariances (1–4 Hz to 58 Hz) mapped to the
  tangent space at a causally updated session reference, plus a causally
  centred < 1 Hz waveform, logistic regression (weight 1/6);
- 5 ShallowConvNets on Euclidean-aligned input, 5 unaligned ShallowConvNets,
  5 aligned FBCNets and 5 unaligned FBCNets with braindecode's fixed FIR
  filter bank (1/30 each);
- r8t4 only: 2 aligned and 2 unaligned FBCNets whose FIR filter bank is
  trained with the network (1/24 each). Learned and fixed filter banks are
  complementary: the learned family helps only alongside the fixed one.

**Test-time adaptation (label-free, causal, reset per session).** Each
window is whitened by the running mean covariance of the session so far,
blended with the training mean as five pseudo-windows. Ensemble
probabilities are divided by the mean predicted class distribution of the
previous 30 windows (blended with a uniform prior of weight five).

**Training data.** PROTEUS only (NEMAR nm000290 v1.0.0, CC BY 4.0), all ten
public participants, official preprocessing. No external data and no
pretrained models in the sealed candidates.

**Selection.** Leave-one-participant-out over the eight non-warm-up-test
participants, the same folds with P04/P09 added as training-only data
(`dev_loso_all`, the sealed models' training regime), and a later-session
probe (P10 sessions 2–3); evaluation on leave-one-participant-out over all
ten (P04/P09 untouched by selection) and later sessions of P04/P10.

**Compute.** Final weights: 24 nets × 1–3 min on one A100, v2 under a minute
on CPU (< 1 GPU-hour). Development: on the order of 100–200 GPU-hours on
the Tübingen ML Cloud (to be summed from `sacct`). Inference on an A100
(worker replica): 25–31 ms per window; CPU fallback 64–83 ms.

**Reproduction.** `python scripts/build_sealed.py scripts/sealed_r8t4.json
OUT` (Galvani: `scripts/cloud/build_sealed.sbatch`; r6nf:
`scripts/sealed_r6nf.json`) rebuilds a package from scratch with the
versions pinned in `requirements/cloud.txt` (neuroai 0.4.0, exca 0.5.29,
braindecode 1.8.1, torch 2.14.1) and checks it against the harness and the
reference package. A clean r6nf rebuild (commit `1631413`) has fixed filter
banks identical to the staged package and differs on 1 of 2,655 windows
(GPU training is not bit-deterministic; audit tolerance stated as such).
`scripts/cloud/worker_replay.sbatch` replays any package in an Apptainer
replica of the official worker image.
