# Handoff: Track 2 state (28 October 2026, night)

Goal: **top three in the sealed phase** (prize money, and the sponsors'
internship and career opportunities). Warm-up rank does not matter. The user
believes top three is reachable and wants a fresh, first-principles look at
the solution space before the sealed phase opens on 5 Nov.

Read this file, then:
- `AGENTS.md` and `OFFICIAL_CONTRACT.md`;
- `docs/SEALED_PLAN.md` (slot plan, method description);
- `docs/OPTIMIZATION_R10.md` (latest round: R10 and R11);
- `docs/STRATEGY_REVIEW.md` §2–3 (solution-space map from 20 Oct; partly
  superseded below).

Older round docs (`OPTIMIZATION*.md`, R1–R8 with R9 inside R8) only when a
number is needed.

## Dates, quotas, rules (recheck live in Codabench before any upload)

| Item | Value |
|---|---|
| Warm-up | open until 1 Nov 2026, 23:59 CET (5/day, 100 total; 7 used) |
| Sealed phase | 5 Nov 00:01 → 21 Nov 23:59 CET; at most 5 uploads, one per day |
| Ranking | best sealed score retained (`Force_Best`); scores visible, logs and outputs hidden |
| Evaluation | ≤ 60 min on one A100 (we keep ≤ 30 min); `submission.py` + data files |
| Audit (top three) | training re-run from the committed config (±2σ) |
| External data | public, redistributable, declared; Scherer 2015 excluded (user) |

**Uploads:** sealed uploads need the user's explicit authorisation each time.
Check quota and phase live first. Warm-up uploads were made by Claude in
Chrome with authorisation.

## Task in one paragraph

Classify three mental tasks: motor imagery, mental subtraction and word
generation.
- **Input:** 4 s, 41 channels, 120 Hz EEG windows (PROTEUS, NEMAR
  nm000290; 10 public participants, two interfaces: Graz, BrainHero).
- **Metric:** balanced accuracy per participant × session × interface cell,
  averaged over cells.
- **Sealed set:** later sessions of the 10 public participants (about 43%)
  and 10 unseen participants with six sessions each (about 57%).
- **Streaming:** `predict(X)` sees one window at a time, in order, with no
  metadata. Every session starts from a fresh copy of the model
  (`2026-competition/benchmark_utils/streaming.py`), so label-free causal
  adaptation works only within a session.

## Where we stand (honest)

All numbers are cell-mean balanced accuracy in %, rebalanced.

| Package | Selection (mean ds, dl_all) | all_loso (P04 / P09) | seen_session | Honest warm-up replay |
|---|---:|---:|---:|---:|
| r9 (staged as r9g/r9b/r9c) | 77.6 (seeds 0-2) | 67.0 (49.9 / 69.2) | 71.5–72.0 | 0.5358 (= Codabench 1014114) |
| R10 = r9 with pretrained SincShallow | +0.7 over 8 seeds | +0.3 | +0.7 (seeds 0-2) | 0.5319 |

- **Within-session ceiling (R8):** our members trained within each
  session reach 81.9, against about 65–67 on unseen participants. About
  17 points sit in participant-specific decision boundaries that we cannot
  learn without labels.
- **Warm-up board (27 Oct, 160 teams).** 52 entries are ≥ 0.90 and 16 are at
  0.60–0.89; these almost certainly trained on P04/P09. 17 are at 0.55–0.59:
  plausible honest entries and the real competition. Our honest 0.536 puts
  us about 20th among plausible honest entries. Caveat: the warm-up has 8
  cells, 6 of them P04, so its noise is several points.
- **Rough expectation:** without a new lever we are a solid mid-to-upper
  entry, not a confident top three. Every lever since R9 moved by ≤ 1 point.

## Candidates and slot plan

Staged in `runs/upload/`; each is worker-replica verified with harness
agreement 1.0:

| ZIP | SHA-256 | Note |
|---|---|---|
| `sealed_candidate_ensemble_all_r9g.zip` | `cd238165…` | r9 clean build from `ea58271`; also ran on the real worker in warm-up |
| `sealed_candidate_ensemble_all_r9b.zip` | `ca47d2b4…` | seed replicate (+5) |
| `sealed_candidate_ensemble_all_r9c.zip` | `c52fb8e5…` | seed replicate (+10) |

**Running at handoff:** clean builds of R10, R10b and R10c from commit
`5ee3adc`.

| Job | Config | Output (in `runs/sealed/` on Galvani) |
|---|---|---|
| 3017691 | `scripts/sealed_r10.json` | `ensemble_all_r10_clean_5ee3adc` |
| 3017692 | `scripts/sealed_r10b.json` | `ensemble_all_r10b_clean_5ee3adc` |
| 3017693 | `scripts/sealed_r10c.json` | `ensemble_all_r10c_clean_5ee3adc` |

The SincShallow members load `runs/pretrain/sincshallow_ea-m7-s{seed}.pt`;
seeds 0–14 exist on Galvani (gitignored).

To finish them:
1. Check `*_check.json`: `agreement_with_harness` 1.0, and the replay's
   `files_unchanged`.
2. Run the worker-image replica (`scripts/cloud/worker_replay.sbatch`; see
   how r9b/r9c were done in `SEALED_PLAN.md`).
3. Copy `submission.zip` to
   `runs/upload/sealed_candidate_ensemble_all_r10{,b,c}.zip`.
4. Record the SHA-256 in `SEALED_PLAN.md`.

**Audit note:** the pretraining checkpoints must be reproducible by the
auditors. That means `scripts/export_moabb.py` (+ `export_moabb.sbatch`),
then `experiments/pretrain.py sincshallow ea m7
physionetmi,cho2017,lee2019_mi,dreyer2023,schirrmeister2017,shin2017a,shin2017b
--seed S`. Shin 2018 word generation came from DepositOnce zips in
`data/shin2018/raw`, but m7 does not use it. Document this in the method
description before shipping R10.

**Proposed sealed order** (in `SEALED_PLAN.md`, not yet confirmed by the
user):

| Day | Package |
|---|---|
| 1 | r9g |
| 2 | R10 |
| 3 | R10b |
| 4 | R10c |
| 5 | Decided from the scores |

Under Force_Best, seed copies of the best recipe are extra draws. A
genuinely better candidate from the next session should replace them.

## Solution space after R10/R11 (step back from here)

**Closed (evidence in the round docs; don't repeat without a new angle):**

| Area | What was tried |
|---|---|
| Input alignment | EA / recentering (k, stretch, train-causal), session RMS, label-shift-aware reference (oracle +0.05) |
| Output adaptation | windowed rebalancing (adopted), Sinkhorn, trial-order prior, AdaBN, time-shift TTA |
| Pseudo-label adaptation | self-training, T3A prototypes, confident-only updates: all reinforce errors |
| Architectures | about 30 braindecode nets aligned and unaligned (R9 sweep); recipes flat around defaults; more seeds; time-segment members; crops |
| Ensemble combination | weights, stacking, calibration/temperature, LDA head, per-session member weighting (oracle) |
| Participant-aware objectives | balanced loss, GroupDRO; R11: SWAD weight averaging, same-class cross-participant mixing |
| Personal models | soft routing by participant recognition: +1 seen, −1 unseen, net ≈ 0 |
| Foundation models | CBraMod and LaBraM fine-tuned, REVE frozen: gain on dev, lose on all_loso / seen_session |
| External data in the 3-class loss | Scherer (licence), Dreyer 2023 (MI-only sessions break EA) |
| Multi-corpus pretraining (R10) | 7–8 open corpora, one head per corpus. Big gains with few training participants (dl +7, ds +9 single nets), shrinking to ≈ 0–3 at 9 participants. Only SincShallow keeps a small ensemble gain |
| Clustering | session-level clustering bound ≈ chance |

**Why progress stalled (first principles):**
1. With 10 training participants, cross-participant variance dominates.
   Methods that add "more of the same" saturate: data, seeds, architectures,
   DG regularisers.
2. Every label-free adaptation needs a signal about the new participant's
   class boundaries. Pseudo-labels at 65% accuracy reinforce errors, and
   cluster structure does not follow the classes.
3. The only past step changes came from **new information**:
   - session alignment (+7);
   - the slow < 4 Hz signal (+7);
   - ensembling heterogeneous input streams (+7);
   - exploiting the balanced runs (rebalancing, +2–6).

**Open directions, ranked by expected value per day (to be challenged):**
1. **More data with the same three tasks.**
   - PROTEUS-like paradigms with motor imagery, mental subtraction and word
     generation in one session: the Inria/Potioc group (PROTEUS is
     "Dreyer2026Proteus"); Friedrich, Scherer and Neuper 2012/2013; BCI
     Competition III-V (MI left/right and word generation in one session,
     3 subjects: check its terms); Shin 2017 A+B (same 29 participants, MI
     and arithmetic).
   - Only multi-task-per-session data can teach the task contrast itself.
     Single-task sessions break per-session alignment, which is why Dreyer
     2023 failed.
   - Search NEMAR, OpenNeuro, Zenodo, BNCI Horizon and MOABB thoroughly.
     Check each licence; skip Scherer per the user.
2. **Unified-label pretraining.**
   - Map external tasks onto the three PROTEUS classes (MI, arithmetic,
     word generation) and pretrain the 3-class head itself, not only the
     trunk.
   - Use per-corpus RMS (raw) alignment, because EA erases task-type means
     in single-task sessions.
   - Check whether the head transfers on `dev_loso_all` and `all_loso`.
3. **Label-free test-time training with a self-supervised auxiliary
   loss.**
   - Masked reconstruction or a contrastive loss on the session's own
     windows, updating normalisation or early layers per session.
   - This avoids pseudo-labels, which is exactly what failed before.
   - It must run within the per-session fresh-copy rule and the 60-min
     A100 limit.
4. **Evaluation power (the user's "bigger validation set").**
   - Decisions are made at ±1-point noise on 8–10 participants.
   - Pool protocols and seeds for every decision: ≥ 6 seeds, report the
     paired difference per participant and a sign test.
   - Treat the honest warm-up replay as an extra proxy for P04/P09 only.
5. **Check the 43% seen-participant part separately.**
   - seen_session is about 72, while ds (P10 only) is about 84.
   - Larger capacity or longer training on all sessions might help seen
     participants without hurting unseen ones.
   - Weight by the sealed composition: about 0.57 × unseen + 0.43 × seen.

## Protocols and rules of evidence (`experiments/protocol.py`)

| Protocol | What it is |
|---|---|
| `dev_loso` | 8 non-test participants, leave one participant out |
| `dev_loso_all` | the same folds with P04/P09 added to training only (the sealed regime) |
| `dev_session` | P10 sessions 2–3 (4 cells, noisy) |
| `all_loso` | all 10 participants left out in turn; only P04/P09 are untouched by selection |
| `seen_session` | later sessions of P04 and P10 |

- **Select** on the mean of `dev_session` and `dev_loso_all` ("sel"), and
  report `dev_loso`.
- **Evaluate** on `all_loso` (P04/P09 reported separately) and
  `seen_session` only after a choice is fixed.
- About ±1 point is noise; prefer plateaus and fresh-seed confirmation.
- Every package needs harness agreement 1.0, an official replay with files
  unchanged, and a worker-image replica run before it is staged.

## How to run things

```bash
# local scoring of saved probabilities (Mac; no training)
.venv/bin/python scripts/mix_probs.py dev_session,dev_loso_all,dev_loso,all_loso \
  v2 'fbcnet_ea_tf:0-1*0.5' 'fbcnet_raw_tf:0-1*0.5' shallow_ea:0-2 shallow_raw:0-2 \
  fbcnet_ea:0-2 fbcnet_raw:0-2 sincshallow_ea_ptm7:0-2 fblight_ea:0-2 \
  --dirs probs_r11,probs_r10,probs_r9,probs,probs_cloud,probs_r5,probs_local,probs_r7,probs_tf,probs_cloud_r8
```

That member list is R10 (seeds 0-2): sel 78.58, all_loso 67.36. r9 has
`sincshallow_ea` instead.

**Galvani (all training runs here).**
- **Connecting:** `ssh mlcloud-galvani`. When 2FA is needed, open a
  terminal tab running `ssh -fN mlcloud-galvani` and ask the user to enter
  the verification code.
- **Workspace:** `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026`.
- **Code sync:** `rsync -az --exclude __pycache__ experiments scripts
  submissions mlcloud-galvani:<ws>/`. `git fetch` fails there, so never
  pull.
- **Partitions:** `--partition=a100-galvani,v100-galvani,2080-galvani
  --gres=gpu:1`. The queue is shared with other users.
- **Sweeps:** `sbatch --array=0-2 [--export=ALL,NI26_PROBS=runs/experiments/probs_X]
  scripts/cloud/sweep.sbatch "cfg1 cfg2" dev_loso,dev_session,all_loso`.
  The array index is the seed.
- **Pretraining and fine-tuning:** `scripts/cloud/pretrain.sbatch ARCH ALIGN
  TAG CORPORA "configs" protocols`.
- **Packages:** `scripts/cloud/build_sealed.sbatch CONFIG.json OUT` with
  `--export=ALL,NI26_SOURCE_REV=<commit>`, then
  `scripts/cloud/worker_replay.sbatch`.
- **Config names** (`experiments/run_cv.py`):
  - `ARCH_ea|raw[_tf]`;
  - `..._ptTAG[_lo|_lp]` loads a pretrained trunk;
  - suffix `_swa` or `_cmix` adds an R11 option.
- **Login node:** it kills long processes. Run exports and heavy work as
  jobs, and keep `pkill -f` patterns out of the ssh command line.

**Mac:** local training only through `scripts/run_capped.sh`, one job at a
time (it froze on 22 Oct). Prefer Galvani.

**zsh:** quote members that contain `*`, and pass lists as arrays.

## Repository state

`main` is at `5ee3adc` (pushed): R10 pretraining, R11 options and configs.
Uncommitted at handoff: this file, `NEXT_SESSION_PROMPT.md`, and the
seeds 6-7 paragraph in `OPTIMIZATION_R10.md`.

Not in git:
- `data/` (PROTEUS export, external corpora, MOABB cache);
- `runs/` (probabilities, packages, ZIPs, pretrain checkpoints).
