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

**Update 27 Oct: r9 leads.** r9 = r8t4 + 5 aligned SincShallowNets + 5
aligned FBLightConvNets (`scripts/sealed_r9.json`, code `ea58271`;
`docs/OPTIMIZATION_R8.md`, R9). Fresh-seed selection 77.71 vs 76.17,
all_loso 66.95 (P04 49.6, P09 69.7) vs 66.31, seen_session 71.99 vs 70.93,
honest warm-up replay 0.5358 vs 0.5304; expected ≈ +0.8 on the sealed set.
Worker replica 51.3 s per 755 windows (r8t4g 46.7 s), ≈ 13 min projected.
New order: **r9 (clean rebuild from `ea58271`), r8t4g, r6nfg**, then up to
two candidates informed by the scores. The r9 warm-up upload
(`warmup11_r9_tv.zip`, 1014114) runs it on the real worker.

**Update 28 Oct: r9 seed copies staged.** Under Force_Best, independent
seed draws of the same recipe are cheap extra chances: each copy has about
±0.4 seed noise, which is worth about +0.3 to +0.4 expected over a single
draw. Both copies were built from `d89f133` on Galvani:

| ZIP | Seeds | Harness agreement | Worker replica (755 windows) | SHA-256 |
|---|---|---:|---:|---|
| `sealed_candidate_ensemble_all_r9b.zip` | +5 (learned-FIR 2–3) | 1.0 | 0.9986, 50.4 s, files unchanged | `ca47d2b4…` |
| `sealed_candidate_ensemble_all_r9c.zip` | +10 (learned-FIR 4–5) | 1.0 | 0.9986, 46.4 s, files unchanged | `c52fb8e5…` |

Both copies use the same `submission.py` as r9g (`396e3866…`). Their
labels agree 100% with the r9 clean build on the official test.

Sealed upload order, one per day:

| Day | Package |
|---|---|
| 1 | r9g |
| 2 | r9b |
| 3 | r9c |
| 4 | r8t4g, or a stronger R10 candidate (pretrained trunks, `docs/OPTIMIZATION_R10.md`) |
| 5 | Decided from the scores |

**Update 28 Oct (evening): R10.** R10 is r9 with the SincShallow family
initialised from trunks pretrained on seven open motor-imagery corpora
(`docs/OPTIMIZATION_R10.md`).

| | Selection | all_loso | seen_session |
|---|---:|---:|---:|
| Gain over r9, six seeds (cross-validation) | +0.9 | +0.5 | +0.7 |

| Honest warm-up replay | Score |
|---|---:|
| R10 | 0.5319 |
| r9 | 0.5358 |

The warm-up difference is within noise: six of its 8 cells are P04, and
pretraining never helped P04. Seed copies R10b and R10c use r9b's and r9c's
seeds.

Proposed order, one per day and every upload authorised by the user:

| Day | Package |
|---|---|
| 1 | r9g (proven on the real worker) |
| 2 | R10 |
| 3 | R19e (user decision 30 Oct; replaces R14) |
| 4 | R19 (user decision 30 Oct; replaces R10b) |
| 5 | R15 (user decision 29 Oct) |

**Superseded by the adaptive order (user decision 30 Oct, evening).**
- Day 1: R19e.
- Day 2: R15.
- Days 3–5: chosen from whether R15 beat R19e on the sealed leaderboard (Force_Best shows the best
  score).

The bootstrap E[max] of {R10, R19e, R19, R15} equals R19e alone, and R15/R19e share worker-proven
code, so r9g and R10 bought neither expected score nor crash insurance (docs/OPTIMIZATION_R10.md,
round 4).

Each R10 package first needs a clean build, harness agreement, a worker
replica run and staging.

**R10b and R10c staged (28 Oct, night).** Clean builds from `5ee3adc`
(Galvani 3017692/3017693), same `submission.py` as r9g (`396e3866…`):

| ZIP | Harness agreement | A100 replay | Worker replica 3017800 (755 windows) | SHA-256 |
|---|---:|---|---|---|
| `sealed_candidate_ensemble_all_r10b.zip` | 1.0 | 0.9986, files unchanged | 0.9986, 49.7 s, files unchanged | `f3ddbe54…` |
| `sealed_candidate_ensemble_all_r10c.zip` | 1.0 | 0.9986, files unchanged | 0.9986, 46.1 s, files unchanged | `51b0570f…` |

R10 (3017691, `sealed_candidate_ensemble_all_r10.zip`, `1e3ab63a…`):
harness agreement 1.0, A100 replay 0.9973 with files unchanged, worker
replica 3017801 0.9973 with files unchanged. It took 99.1 s for 755 windows,
but it ran on galvani-cn222, which was about twice as slow throughout (the
build took 61 min against 25 min). R10 has the same members and
`submission.py` as R10b/R10c (47–50 s on cn201) and r9 (30.5 s on the real
worker).

**Real worker (29 Oct):** the exact staged R10 ZIP ran in the warm-up phase
as Codabench 1018364: 0.99731 cell-mean (identical to the replica) in 30.8 s.
(The submissions table kept showing "Running"; `/api/submissions/<id>/`
had the result.) Warm-up quota after it: 8 of 100.

**R14 staged (29 Oct).** R10 with random-rotation augmentation on the
aligned FBCNet, SincShallowNet and FBLight members (docs/OPTIMIZATION_R10.md
R14; selection 79.31 vs 78.58 on seeds 0-2, 77.92 vs 77.69 on seeds 3-5).
Clean build from `741a074` (Galvani 3023343, `scripts/sealed_r14.json`),
same `submission.py` (`396e3866…`): harness agreement 1.0; A100 replay
0.9960, files unchanged; worker replica 3023344 0.9960, 49.0 s, files
unchanged. `sealed_candidate_ensemble_all_r14.zip` (`0ca4df88…`). Against
the R10 package: 1 of 755 official-test labels and 6 of 2,655 windows
differ. Day 3 is R14 (user decision 29 Oct); R10b moves to day 4.

**R15 staged (29 Oct).** R14 with SincShallowNet trunks pretrained with
rotation (`sincshallow_ea-m7r-s{0-4}.pt`; docs/OPTIMIZATION_R10.md R15;
selection 79.59 vs 79.31 on seeds 0-2, 78.23 vs 77.92 on seeds 3-5). Clean
build from `6a894df` (Galvani 3023595, `scripts/sealed_r15.json`), same
`submission.py` (`396e3866…`): harness agreement 1.0; A100 replay 0.9960,
files unchanged; worker replica 3023596 0.9960, 49.7 s, files unchanged.
`sealed_candidate_ensemble_all_r15.zip` (`0f86ccbd…`). Against R14, only
the five SincShallowNet members change; 0 of 2,655 labels differ. Day 5
(user decision 29 Oct).

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

**R19 staged (30 Oct).** R15's weights, unchanged, plus the per-session
gaze-template stage (docs/OPTIMIZATION_R10.md R19; `scripts/add_session_em.py`
on `ensemble_all_r15_clean_6a894df`, Galvani 3024078). There is a **new
`submission.py`** (`55d11fbf…`) with the `em_*` keys. Checks on all 2,655
windows:
- with the stage off, the new code reproduces R15 exactly (1.0);
- with it on, it equals experiments/session_em.py on the package's own
  rebalanced mix (1.0).

Worker replica 3024079: 0.9960, 50.2 s, files unchanged.
`sealed_candidate_ensemble_all_r19.zip` (`3db3a1fc…`).

**Real worker (30 Oct):** the exact staged R19 ZIP ran in the warm-up phase
(30961) as Codabench 1021075, with user authorisation: 0.99601 cell-mean
(identical to the replica) in 30.8 s. Warm-up quota after it: 9 of 100.
Live check the same day: sealed phase 30483 runs 4 Nov 23:01 UTC to 21 Nov
22:59 UTC, 1 per day, 5 in total.
Audit config: `scripts/sealed_r19.json` (R15 members plus `session_em`);
`build_sealed.py` reproduces the stage from it.

**Day 4 is R19 (user decision 30 Oct).** It replaces R10b, which differs
from R10 on only a few windows. Only the best of the five sealed scores
counts, so having both R15 (day 5) and R19 (day 4) covers both outcomes of
the gaze stage. R10b stays staged as a fallback.

**R19e staged (30 Oct).** R15's weights unchanged, plus the per-session EM
stage with 32 gaze values and 16 ocular-event rates
(docs/OPTIMIZATION_R10.md R19e; `scripts/add_session_em.py --events` on
`ensemble_all_r15_clean_6a894df`, Galvani 3025576). New `submission.py`
(`6c63a8f2…`) with the `em_ev_*` keys. On all 2,655 windows:
- stage off reproduces R15 exactly (1.0);
- stage on equals experiments/session_em.py on the package's own
  rebalanced mix (1.0).

Worker replica 3025577: 0.9869, 50.0 s, files unchanged (an in-sample
score, not evidence). `sealed_candidate_ensemble_all_r19e.zip`
(`73cd31f7…`). Audit config: `scripts/sealed_r19e.json`.

**Real worker (30 Oct):** the exact staged R19e ZIP ran in the warm-up phase
(30961) as Codabench 1021330, with user authorisation: 0.98685 cell-mean
(identical to the replica) in 31.4 s. Warm-up quota after it: 10 of 100.

**Fresh-clone audit of R19e (30 Oct, night).** A clean clone of `d26234f` (submodule `a66bc23`) on
Galvani rebuilt everything from public sources:
- the PROTEUS windows (3027901), the 7 corpora re-downloaded (3027927–33), the m7r trunks (3027934);
- then `build_sealed.sbatch scripts/sealed_r19e.json` (3027935).

Results:
- **Corpora:** content equal to the originals (3029111). X differs by at most 3e-5 µV; y, subject,
  session, run, task and split are identical.
- **Trunks:** relative distance to the originals 0.05; two original seeds differ by 0.13–0.16.
- **Package:** `submission.py` identical (`6c63a8f2…`). Labels equal the staged R19e on 755/755
  official-test windows and on 2,654 of 2,655 windows overall (max probability diff 0.031). That is
  the same tolerance as the r6nf clean rebuild.
- **A100 replay:** 0.9882 in-sample (staged 0.9869), 71.6 s, files unchanged.

Two gaps found and closed:
- `build_sealed.sbatch` now creates `data/proteus_windows.npz` on a fresh clone. `export_moabb.py`
  also needs that file, so export it first.
- `scripts/sealed_r19e_inputs.sha256` records the input hashes and the export chunk sizes. Other
  chunk sizes change the file hashes and the pretraining row order, not the content.

**Day 3 is R19e (user decision 30 Oct).** It replaces R14, which gets the
same answer as R15 on all 2,655 public windows. Order: r9g, R10, R19e, R19,
R15. The gaze stage is covered at two strengths (R19, R19e), with R15 as
the no-stage fallback. R14 stays staged as a fallback.

## Pre-freeze audit and guarded packages (31 Oct)

Three independent reviews (package code, evaluation method, strategy) plus a
stress test of the staged R19e package on real windows
(`scripts/diag/stress_package.py`).

**Defects found in the staged packages (R19, R19e; R15/R10 unaffected unless noted):**
- **Crash (lost slot, no log):** `SessionEM` inverts a zero covariance when every
  EM feature is constant over the session's past. That happens when the first
  ≥ 10 windows are all-NaN or zero (one NaN in a raw run, spread by the official
  filter, would do it) or when the 12 EM-view channels are dead or identical.
  The public data have no NaN and no flat channel, so it is unlikely, not impossible.
- **Silent numerical bug (also in `experiments/session_em.py`):** `sd = std + 1e-9`.
  An event-rate column that never fired is constant in the past; its mean is off by
  one ulp, so a later window with an event gets z ≈ 2e9 and rounding noise decides
  the class (28 % of EM steps had a constant column). Fixed with an sd floor of 1e-3.
  Six-seed dev check (`scripts/diag/check_sd_floor.py`, `runs/diag/check_sd_floor.log`):
  R19 unchanged everywhere; R19e dla +0.05 / +0.10 (seed halves, ≤ 2 labels), ds +0.00.
  The bug did not distort the dev evidence.
- **NaN poisoning (all packages):** one non-finite member probability entered the
  rebalance history and EM state and spoiled the rest of the session (needs ≥ 1e25 µV).
- **Positional channels (all packages):** `meta["ch_names"]` was unused, and the
  organisers' guide says sealed runtime `meta` values may differ. A reordered
  montage would have permuted every model's input silently.

**Fixes (`submissions/ensemble/submission.py`, `cdfa4cec…`; reference code in step):**
EM fit inside try/except with a finite check (falls back to the rebalanced
posterior), sd floor 1e-3, non-finite member rows set to uniform before any
session state, and `channel_map`: inputs are re-ordered by `meta["ch_names"]`
against the hard-coded training montage (missing channels zero-filled; identity,
absent or unusable names keep the positional path). On normal data the guards are
inert except the sd floor.

**Checks:**
- `add_session_em.py` from the R15 package: with the new code, stage off reproduces
  R15 exactly (1.0) and stage on equals `experiments/session_em.py` (1.0), for R19
  and R19e. Weights stay the staged ones (R19e `1b6c5d03…`, R19 `5e720d63…`, R15
  from `0f86ccbd…`), so the audit configs are unchanged.
- Stress test: every degenerate case above now runs; a permuted, lower-case or
  40-channel montage with matching `ch_names` gives labels identical to the original.
- Official CPU replay of R19e-g: 0.99080, 41 s, files unchanged.
- Worker replica 3031525: R19e-g 0.99080 (51.7 s), R15-g 0.99601 (46.6 s),
  R19-g 0.99601 (50.8 s), files unchanged.

| ZIP (`runs/upload/`) | SHA-256 |
|---|---|
| `sealed_candidate_ensemble_all_r19e_g.zip` | `125503e8f59b…` |
| `sealed_candidate_ensemble_all_r15_g.zip` | `e10409ba1b1c…` |
| `sealed_candidate_ensemble_all_r19_g.zip` | `c10650cad4ed…` |

**Real worker (31 Oct):** the exact staged R19e-g ZIP ran in the warm-up phase
(30961) as Codabench 1024097, with user authorisation: 0.99080 cell-mean
(identical to the replica) in 31.7 s. Warm-up quota after it: 11 of 100.
The guarded `submission.py` (`cdfa4cec…`) is now worker-proven; R15-g and R19-g
run a subset of its code path. The `_g` ZIPs replace the unguarded R19e/R15/R19
ZIPs. Code is frozen from here: new sealed arms may only change `weights.npz`.

**Evidence caveats from the reviews (not new levers):**
- P10 carries about 65 % of the selection rule (all of ds, 6 of 20 dla cells).
- R19e − R15 on dla per participant (six seeds): mean +2.67, bootstrap 95 % CI
  [+0.68, +4.35], 7/8 positive; R19 − R15 +1.44 [+0.44, +2.44]; R19e − R19 +1.23
  [−0.35, +2.68], within selection-bias range. Expected sealed gain of R19e over
  R15 after shrinkage: about +0.6 to +1.0.
- The sealed seen/new split is not published (FAQ: identities and session
  allocation private). 43 % seen assumed all six sessions; carry 33–50 %.
- P04/P09 (al) and seen_session have been looked at during selection; they are no
  longer clean holdouts.
- Sealed composite expectation: about 69.5–73.

**Sealed order revised (user decision 31 Oct).** The branch for "R15 beats R19e"
now uses R19-g and R19e-λ0.25-g instead of R14, R10 and r9g, which agree with R15
on all but 0–6 of 2,655 public labels and add about 0 to the best-of-five.
R19e-λ0.25-g: R15 weights plus the R19e stage at λ 0.25
(`scripts/add_session_em.py --events --lam 0.25`, audit config
`scripts/sealed_r19e_l25.json`); stage off reproduces R15 (1.0), stage on equals
`experiments/session_em.py` at λ 0.25 (1.0); same worker-proven `submission.py`
(`cdfa4cec…`). Dev (six seeds): dla 73.52, ds 85.76, al 68.15 (R19 73.57 / 85.68 /
68.17; profile in docs/HANDOFF.md). Worker replica 3031717 (A100): 0.99601, 49.8 s,
files unchanged; `sealed_candidate_ensemble_all_r19e_l25_g.zip` (`b65086f8e0fe…`).
(3031715 failed before scoring: Slurm placed it on a non-A100 GPU, which the worker
image's CUDA build does not support; replicas must use the default A100 partition.)
Day 1 R19e-g, day 2 R15-g; slots 3–5 may wait
for weights-only arms until about 18 Nov.

## Model-file member before the code freeze (31 Oct, night)

After the warm-up closes, sealed packages may only change data files. To keep
new architectures possible after the freeze, `submission.py` gained a generic
`Jit` member: a net whose dict has `jit_file` loads a TorchScript module from the
submission folder (aligned (B, C, T) float32 → logits (B, 3)) and keeps its
input stream through the usual alignment keys. It is stateless and shared by the
per-session deep copies. Packages without `jit_file` take exactly the old path.

**Proof package:** `scripts/export_jit_member.py` swaps R19e-g's member 0
(ShallowConvNet) for a scripted twin (`member0.pt`, exported with torch 2.14.1,
the worker's version). On all 2,655 public windows the twin's probabilities equal
Shallow's (max |Δp| 0.0) and the package's labels equal R19e-g's (1.0). Official
CPU replay: 0.99080 (= R19e-g), 42 s, `member0.pt` at the ZIP root, files
unchanged. `scripts/replay_package.py` now ships `.pt` files.

Worker replica 3032473 (A100): 0.99080, 51.6 s, files unchanged
(`warmup13_r19e_jit_g.zip`, `011e56bc77b4…`). **Real worker (1 Nov, 00:06 CET):**
the exact ZIP ran in the warm-up phase as Codabench 1024447, with user
authorisation: 0.99080 (identical to R19e-g) in 31.8 s. Warm-up quota after it:
12 of 100. This `submission.py` (`25d51a6e…`) is the frozen sealed code: it runs
every existing package unchanged and accepts TorchScript members as data files.
Export members with torch 2.14.1 (the worker's version) and prove each with
`export_jit_member.py`-style equivalence plus a worker replica before staging.

## External-head teacher: MI-transfer arm (1 Nov, candidate)

**Idea.** MI is the largest loss block, and the only class with hundreds of
external participants. The m7r pretrained nets (`experiments/pretrain.py`, seven
public MI corpora, about 340 participants, no PROTEUS data) are kept frozen. Their
per-corpus head log-probs on the causal-EA PROTEUS stream feed a logistic map to
the three classes (C 0.1, balanced; fit per fold on training participants only).
The teacher is stateless, so it ships as one TorchScript member (data files only,
frozen code).

**Screen** (`scripts/diag/screen_external_heads.py`, no PROTEUS training in the
nets; R19e from the six-seed base cache):
- MI vs SUB/WG from High-Gamma `1 − p(rest)`: mean AUC 0.70–0.77, P10 0.81–0.85.
- PhysionetMI rest, Shin 2017B subtraction and |laterality| are near 0.5.
- The teacher alone reaches dla 52–54. P(right | R19e wrong) is 0.45–0.50,
  against 0.23–0.29 for the round-4 views.

**As an R19e member** (`scripts/diag/eval_external_teacher.py`): it enters the
member mix with weight w (one family = 1), then the unchanged rebalancing and
R19e stage. Seed halves use the matching pretraining seeds. Teacher sinc+fbcnet,
sel Δ:

| w | s0–2 | s3–5 | pooled s0–5 | dla pooled | al pooled |
|---|---:|---:|---:|---:|---:|
| 1 (pre-registered) | +0.79 | +0.94 | +0.29 | 74.82 → 75.59 | 69.23 → 69.62 |
| 2 | +0.91 | +1.36 | **+0.84** | 74.82 → 76.43 | 69.23 → 70.26 |

- Pooled w2, dla per participant: 6 of 8 positive (P10 +3.4, P02 +2.6, P05 +2.1,
  P03 −1.6).
- sinc-only and fbcnet-only teachers behave the same (pooled w2 +0.86 / +0.81).

**Control.** The same map on randomly initialised trunks of the same shapes gives
sel Δ −0.41 to +0.29 (teacher alone 32–39). The gain comes from the external
pretraining.

**Caveats.**
- w 2 was chosen after seeing three weights; the pre-registered w 1 misses the
  pooled +0.5 bar.
- The official test participants, evaluation only, move P04 −1.1…+0.7 and
  P09 −0.5…−3.2 (al is still +0.4…+1.0).
- ds is P10 only.
- Treat it as a portfolio arm for days 3–5, not day 1.

**Package (staged 1 Nov, `runs/sealed/final_r19e_ext_g`).** `scripts/export_external_teacher.py`
adds `teacher.pt` (the 12 frozen nets, seed-averaged head log-probs, the map fit on all
public windows) to R19e-g with member weight 2 × 0.125, sharing the EA input stream
(k 5, shrink 0.001). Frozen `submission.py` `25d51a6e…`; only data files change.

- Tracing fixes: braindecode properties that raise `ValueError` get metadata values
  (forward unchanged), and same-device `.to(x.device)` calls are skipped during the
  trace. Without the second fix the graph pinned the sinc and filter-bank filters to
  the CPU, which would have failed on the CUDA worker; the export now asserts that no
  device constant is left.
- Checks (Galvani 3034483, `final_r19e_ext_g_check.json`): traced = eager (0.0; batch
  1 8e-7; loaded on CUDA 2e-6); package = harness reference on all 2,655 windows; 18
  labels differ from R19e-g.
- Official replay on A100 (3034485): files unchanged, 136.5 s. Worker-image replica
  (3034484, default A100 partition): files unchanged, 70.0 s.
- In-sample score 0.98299 (R19e-g 0.99080). Not evidence: every package trains on
  these participants.
- ZIP `runs/upload/sealed_candidate_ensemble_all_r19e_ext_g.zip`, SHA-256 `6562d2ee…`
  (byte-identical to the replica's).
- **Real worker (1 Nov, user-authorised warm-up upload):** Codabench 1026185, Finished,
  0.9829861111 (identical to the replica on all three scores), 44.6 s. Live check
  before it: warm-up phase 30961 current, 12 of 100 used. TorchScript braindecode nets
  (sinc and filter-bank layers) therefore run on the real CUDA worker.
- Audit: still needs a committed config that rebuilds it (pretrain.py m7r s0–5 for
  both families, then the exporter).

**Wave-2 teacher screen (1 Nov, Galvani 3034726, `eval_wave2.json`).** Same map and
folds, eight head families, six pretraining seeds each. Adoption rule, set before the
data: beat ref (the packaged teacher) by ≥ +0.5 pooled sel at its best weight, both seed
halves positive. Pooled sel Δ vs R19e at w 0.5 / 1 / 2:

| family | w0.5 | w1 | w2 | vs ref (best w) |
|---|---|---|---|---|
| ref: sinc+fbcnet m7r (packaged) | +0.23 | +0.29 | +0.84 | (reproduces) |
| m9r: + shin2018wg corpus | +0.07 | +0.20 | +0.72 | −0.12 |
| u9: unified MI/SUB/WG head | −0.21 | −0.29 | −0.98 | worse |
| m7r4: + fblight + shallow m7r | +0.25 | +0.82 | +1.11 | +0.27 |
| fbl: fblight m7r alone | +0.07 | +0.22 | +0.33 | −0.51 |
| sh: shallow m7r alone | +0.27 | +0.44 | +0.52 | −0.32 |
| mix: ref + m9r | +0.01 | +0.28 | +0.37 | −0.47 |
| ref_u9: ref + u9 | +0.03 | +0.40 | +0.67 | −0.17 |

No family passes. m7r4 is the only one ahead (dla 76.49 vs 76.43 at w 2; ds 87.52 vs
87.04, P10 only), within noise. Task-matched corpora (word generation, unified head) do
not help a frozen-head teacher; more MI architectures help a little. The packaged
teacher stays. The remaining bet in this class is end-to-end fine-tuning on PROTEUS,
not more frozen heads.
