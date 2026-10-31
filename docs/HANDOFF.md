# Handoff: Track 2 state (30 October 2026, evening)

Goal: **confidently top three in the sealed phase** (prize money, the
sponsors' internship and career opportunities). Warm-up rank does not
matter. The user wants a clean thread that:
1. understands the current state;
2. does at least 30 minutes of wall-clock state-of-the-art and
   interdisciplinary research;
3. then keeps iterating towards the next real breakthrough, not narrow
   knob-turning.

Read this file, then:
- `AGENTS.md`, `OFFICIAL_CONTRACT.md`;
- `docs/SEALED_PLAN.md` (slot plan, packages, real-worker runs);
- `docs/OPTIMIZATION_R10.md` (R10–R19e, rounds 2 and 3; the latest
  evidence is at the end);
- `docs/GROUND_UP_REVIEW.md` (first-principles review of 30 Oct: bounds,
  solution-space map, literature).

Older round docs (`OPTIMIZATION*.md` R1–R8, `STRATEGY_*.md`) only for a
specific number.

## Dates, quotas, rules (checked live on Codabench 30 Oct; recheck before any upload)

| Item | Value |
|---|---|
| Warm-up ("Warm-up Phase (update 1)", phase 30961) | open until 1 Nov 2026 23:59 CET; 5/day, 100 total; **10 used** |
| Sealed ("Sealed Final Phase", phase 30483) | 5 Nov 00:01 → 21 Nov 23:59 CET; **1/day, 5 total**; best score counts (Force_Best); logs hidden |
| Evaluation | ≤ 60 min on one A100; `submission.py` + data files in a ZIP |
| Audit (top three) | training re-run from the committed config (±2σ); every package has a committed `scripts/sealed_*.json` |
| External data | public, redistributable, declared. Scherer 2015 is excluded by user decision (CC BY-NC-ND). Upstream main (2b53aec, 26 Oct) now reads "Public datasets may be used when their original terms permit competition use"; the pinned a66bc23 is unchanged. Check the live Terms page before relying on either |
| Codabench | competition 17982, account `s_boehler`. Uploads go through Claude in Chrome, only with explicit user authorisation per upload |

## Task in one paragraph

Classify three mental tasks: 0 motor imagery (MI), 1 mental subtraction
(SUB), 2 word generation (WG).
- **Input:** 4 s windows, 41 fronto-centro-parietal channels (M1 reference,
  no EOG/EMG), 120 Hz. Windows run from cue +1 to +5 s, and feedback starts
  at +3 s. PROTEUS has 10 public participants and two interfaces (Graz,
  BrainHero).
- **Metric:** balanced accuracy per participant × session × interface cell,
  averaged over cells. Runs are class-balanced 10/10/10.
- **Sealed set:** later sessions of the 10 public participants plus about
  10 new participants. The split is not published; 43% seen assumed all six
  sessions, so carry 33–50% (31 Oct review).
- **Streaming:** `predict(X)` sees one window at a time, in order, with no
  metadata. Each session starts from a fresh model copy, so label-free
  causal adaptation works only within a session.

## Where we stand

Numbers are cell-mean balanced accuracy (%), from six-seed
saved-probability replays.
- **ds** (`dev_session`): P10 sessions 2–3, seen-participant proxy.
- **dla** (`dev_loso_all`): leave-one-participant-out without the P04/P09
  folds, unseen proxy.
- **al** (`all_loso`) and **ss** (`seen_session`, P04 + P10): diagnostics,
  never used for selection.

| | ds | dla | al | ss (R15 members) |
|---|---:|---:|---:|---:|
| R15 (ensemble) | 85.25 | 72.11 | 67.03 | 71.10 |
| R19 (+ per-session gaze EM) | ≈ 85.4 | ≈ 73.5 | ≈ 68.2 | 71.41 |
| **R19e** (+ ocular-event rates) | **≈ 86.8** | **≈ 74.4** | **≈ 68.9** | **72.35** |
| Within-session ceiling (own labels) | 88.9 | 85.4 | 81.9 | – |

- **Noise:** about ±0.7 seed noise on sel = mean(ds, dla); participant
  bootstrap sd 2.6.
- **Adoption rule:** ≥ +0.5 pooled over six seeds, paired per-participant
  signs, positive on both seed halves (0–2, 3–5).
- **Sealed expectation:** composite about 70–75. Top three is plausible but
  not assured.
- **Warm-up board:** our honest replay (0.536) sat below the plausibly
  honest cluster (0.55–0.59). That board has 8 cells, 6 of them P04, our
  weakest participant (about 49 for every one of our ~200 variants).

## What R15 / R19e are

- **R15:** an ensemble whose rebalanced, weighted probabilities are
  averaged:
  - v2 filter-bank tangent-space LR (weight 1/8);
  - ShallowConvNet EA and raw;
  - FBCNet EA (rotation 0.05) and raw;
  - FBCNet with trainable FIR, EA and raw;
  - SincShallowNet EA with a rotation-pretrained trunk (m7r: 7 open MI
    corpora);
  - FBLight EA (rotation 0.05).

  Inputs use causal Euclidean alignment, and the output uses causal
  windowed rebalancing (W30, α1, k5).
- **R19:** R15 plus a per-session causal soft-EM stage
  (`experiments/session_em.py`). Gaussian class templates are fitted on 32
  slow frontal "gaze" features, seeded by the posterior, with class means
  shrunk to the session mean. The stage is combined as a product of
  experts with exponent 0.5·n/(n+60) from window 10.
- **R19e:** R19 plus 16 causal, sign-free ocular-event rates (`EventRates`:
  8 Hz low-passed derivatives, thresholds 4× and 8× the running median,
  4 bins), giving a 48-dimensional view.
- **Packaging:** `submission.py` contains the stage, enabled by
  `em_*`/`em_ev_*` keys in `weights.npz`. `scripts/add_session_em.py
  [--events]` adds it to an existing package and checks it (stage off
  reproduces the source; stage on matches the reference).

## Packages (in `runs/upload/`, all verified)

| ZIP | SHA-256 | Real worker |
|---|---|---|
| `sealed_candidate_ensemble_all_r9g.zip` | `cd238165…` | warm-up run 1014114 (r9) |
| `sealed_candidate_ensemble_all_r10.zip` | `1e3ab63a…` | 1018364: 0.99731, 30.8 s |
| `sealed_candidate_ensemble_all_r14.zip` | `0ca4df88…` | replica only (same code as R10) |
| `sealed_candidate_ensemble_all_r15.zip` | `0f86ccbd…` | replica only (same code as R10) |
| `sealed_candidate_ensemble_all_r19.zip` | `3db3a1fc…` | 1021075: 0.99601, 30.8 s |
| `sealed_candidate_ensemble_all_r19e.zip` | `73cd31f7…` | 1021330: 0.98685, 31.4 s |
| `sealed_candidate_ensemble_all_r19e_g.zip` | `125503e8…` | replica 3031525: 0.99080 (guarded code) |
| `sealed_candidate_ensemble_all_r15_g.zip` | `e10409ba…` | replica 3031525: 0.99601 (guarded code) |
| `sealed_candidate_ensemble_all_r19_g.zip` | `c10650ca…` | replica 3031525: 0.99601 (guarded code) |

**Guarded code (31 Oct, `submission.py` `cdfa4cec…`).** The pre-freeze audit found
a crash path in the EM stage, an sd bug, NaN poisoning and positional channels;
all fixed, weights unchanged (docs/SEALED_PLAN.md, "Pre-freeze audit"). The `_g`
ZIPs supersede the unguarded R19e/R15/R19 once the guarded code has run on the
real worker. After the warm-up closes (1 Nov 23:59 CET) the code is frozen: new
sealed arms may change only `weights.npz`.

**Sealed order: adaptive (user decision 30 Oct, evening).** It replaces r9g, R10, R19e, R19, R15.
One upload per day from 5 Nov.
- **Day 1:** R19e. It is the best package and has run on the real worker (1021330).
- **Day 2:** R15, the same weights without the stage, on R10's proven code. Under Force_Best, the
  leaderboard shows whether R15 beat R19e on the sealed set.
- **Days 3–5:** decided from those two results.
  - If R19e ≥ R15, the stage helps: use stage variants or the EA-clip build if it passes its gate.
  - If R15 > R19e, the stage hurts on sealed data: use non-stage arms (R14, R10, r9g).

r9g and R10 add no crash insurance (R15 and R19e share proven code) and about 0 to E[max]
(docs/OPTIMIZATION_R10.md, round 4).

**Warm-up scores are not evidence.** Every package trains on the warm-up
test participants, so they score about 0.99.

## The bottleneck (established, with numbers)

The loss is **participant-specific class directions** with only 10
training participants.
- **Gap to the within-session ceiling on the same rows:** 13.3 on dla,
  split MI 5.5, SUB 3.2, WG 1.9.
- **Largest blocks:**
  - P10: MI recall about 30 against 94 with its own labels;
  - P03: all classes;
  - P05 and P07: SUB;
  - P04: weak signal overall. It is not low amplitude; that earlier claim
    was corrected.

**Ruled out as causes:** bias, reference, amplitude, interface, position,
member routing.

**Round 3 (30 Oct) oracles.** These use true labels of earlier windows of
the same session, causally.
- **What labels could buy:** per-session adaptation has capacity of up to
  +5.4 dla over R19e (ocular templates; head refit +4.7).
- **What label-free teachers give:** every one gives about 0. R19e's
  errors are participant-structured. Random labels at R19e's own error
  rate would still be worth +1.4 to +2.3, while R19e's own labels are
  worth 0.
- **No independent label source in these 41 channels:**
  - a permutation oracle on the templates gives +0.03;
  - cardiac, respiration and EMG views make the same errors as R19e.

## Closed levers (do not re-propose without new evidence)

Details and numbers are in `docs/OPTIMIZATION_R10.md` and
`docs/GROUND_UP_REVIEW.md` §4–5.
- **Models and training:**
  - architectures (~30 braindecode nets) and recipes;
  - more seeds;
  - DG objectives (DANN, MixStyle, DRO, SWAD, mixing);
  - crops and time segments;
  - participant bagging;
  - label-noise reweighting.
- **Test-time adaptation:**
  - pseudo-label, entropy, BN and T-TIME;
  - per-session refit (LR/MAP, adaptive LDA on own pseudo-labels);
  - Dawid-Skene member confusions;
  - cortical discriminant-subspace EM;
  - head refit as a proxy for test-time fine-tuning.
- **Output level:** bias/affine maps, exact counts, Graz order prior,
  member weighting, position schedules, interface-switch resets.
- **Alignment variants:**
  - band-limited EA (R19 closed);
  - log-Euclidean mean (+0.25 proxy);
  - RMS and scalar normalisation;
  - label-shift-aware reference.
- **Pretraining and external data:**
  - foundation models (CBraMod, LaBraM, REVE);
  - R17 unified rest-anchored pretraining;
  - adding mental-task corpora to the m7 corpus mix (Shin 2017B/2018WG:
    ptmt, ptm9);
  - FBCNet/FBLight m7r (R16, +0.3, not adopted).
- **Seen-participant routing:** fingerprint, decodability, gaze library;
  0/4 correct at window 30, even with oracle labels.
- **New signal sources:**
  - high-frequency / EMG (30–58 Hz) at chance;
  - imaginary coherency (refuted);
  - amplitude/slow-block scaling;
  - population-template priors in the gaze EM (−0.6).
- **Round 4 (30 Oct evening; `docs/OPTIMIZATION_R10.md`, end):**
  - unsupervised per-session clustering, with or without drift removal;
  - 2-way splits inside predicted classes, even with an oracle relabel;
  - sign-free or signed cortical band power;
  - the feedback-period teacher;
  - posterior ERP and alpha lateralisation (lane views);
  - signed saccade events;
  - mirror (handedness) augmentation;
  - direction-consistency feature gating;
  - per-interface and per-run-type EM;
  - EM λ / recency / feature clipping;
  - the EA reference trace clip (R20: dla +0.08, ds −0.13).

  The ensemble names every class right: the loss is the partition, not a swap. Every view at ≥ 45 %
  alone repeats R19e's errors; every view with independent errors is at 36–39 %, below the 50 % bar.

## What remains open (honest)

**Round 4 status.** The research block plus about 20 oracle screens found no lever worth +1 dla.
The literature agrees: SPDAlign gets 72.7 label-free against 73.2 for EA, and in-context models need
labelled context. What is left is sealed-phase work:
- the upload order (adaptive: R19e, then R15);
- audit reproducibility: done. A fresh-clone rebuild of R19e from re-downloaded corpora matches the
  staged package on 2,654 of 2,655 labels (docs/SEALED_PLAN.md; checksums in
  `scripts/sealed_r19e_inputs.sha256`);
- an EA artifact clip: tested as R20, dla +0.08, not adopted (closed);
- replacing R10 with an R19e variant (E[max] +0.03 to +0.3).

Details are in docs/OPTIMIZATION_R10.md, end.

1. **Base models whose class directions generalise across people.** This
   is the population-level transfer gap (dla 74 against 85). It needs:
   - more same-task participants: no public corpus has all three tasks;
     the Scherer licence decision stays with the user;
   - or a fundamentally better inductive bias. The literature shows no
     ≥ +0.5 gain over an aligned, rebalanced, heterogeneous ensemble, but
     the next session should search harder: 2025–26 work, other fields,
     task-specific neuroscience.
2. **A label source with errors independent of the ensemble.** It must
   beat the "random-error" bar (+1.4 to +2.3) in the ocular-template
   machinery. Physiological views tried so far fail.
3. **Portfolio:** fill the R10 slot with the best decorrelated arm. Only
   the best of the five sealed scores counts.
4. **Seen participants' later sessions (43% of cells):** ds 87 against a
   ceiling of 89. Proxies disagree (ss 72), and sessions 4–6 are further
   in time than anything we can measure.

## Infrastructure

**Galvani (all training here).**
- **Connecting:** `ssh mlcloud-galvani`. When 2FA is needed (the master
  connection drops after about 1 h), open a terminal tab running
  `ssh -fN mlcloud-galvani` and ask the user for the verification code.
- **Workspace:** `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026`.
  Sync code with `rsync -az --exclude __pycache__ experiments scripts
  submissions mlcloud-galvani:<ws>/`. Never `git pull` there.
- **Partitions:** `--partition=a100-galvani,v100-galvani,2080-galvani
  --gres=gpu:1`.
- **Sweeps:** `sbatch --array=0-2 --export=ALL,NI26_PROBS=runs/experiments/probs_X
  scripts/cloud/sweep.sbatch "cfg1 cfg2" dev_session,all_loso` (the array
  index is the seed). Configs are in `experiments/run_cv.py`:
  `ARCH_ea|raw[_tf][_ptTAG][_rot05]...`, and MODS suffixes stack.
- **Pretraining:** `scripts/cloud/pretrain.sbatch ARCH ALIGN TAG CORPORA
  'cfgs' protos 'extra'`.
- **Packages:**
  - `scripts/cloud/build_sealed.sbatch CONFIG.json OUT` with
    `--export=ALL,NI26_SOURCE_REV=<commit>`;
  - a stage on an existing package: `sbatch --wrap 'python
    scripts/add_session_em.py SRC OUT [--events]'`;
  - then `scripts/cloud/worker_replay.sbatch PKG` (Apptainer replica of
    the worker; its `submission.zip` is the upload ZIP).
- **Waiting on jobs:** use a background Bash loop on `squeue`, not
  foreground sleeps.

**Mac (36 GB shared RAM; froze on 22 Oct).**
- Every Python job that loads data runs through `NI26_MAX_GB=6
  scripts/run_capped.sh LOG .venv/bin/python SCRIPT`. It serialises jobs
  and enforces the memory cap.
- Always use `.venv/bin/python`: base miniconda's scipy is broken.
- No local training.

**Saved-probability diagnostics.**
- `scripts/diag/diag_common.py` loads six-seed member probabilities for
  any protocol: `load(proto, spec)`, `mixp`, `rebal`, `cells`, `cm`.
  The probabilities live under `runs/experiments/probs_*` (gitignored, on
  disk; `probs_r15ss` holds the seen_session files for R15's rotation
  members).
- `scripts/diag/verify_r19e.py` is the template for a pre-registered check
  of a per-session stage: seed halves, nulls, nested selection,
  truncation. Smoke test: R15 dla 72.11.
- Round 2/3 scratch evidence (scripts, logs, caches) is in
  `/private/tmp/claude-501/-Users-sebastianboehler--claude-spares-spare-Pdn7MS/f1aede87-a130-4f66-8c95-5c7dc01a3281/scratchpad/`
  (`round2/`, `round3/`, `neuro2/`). It is not guaranteed to survive a
  reboot.

**Codabench:** use Claude in Chrome; if it is disconnected, ask the user to
reconnect. In the page, `/api/competitions/17982/` gives phases and
`/api/submissions/?phase=<id>&owner=s_boehler` gives usage. Upload through
the file input on the participate tab, then poll
`/api/submissions/<id>/`.

## Repository state

`main` is at `954904b` (pushed). Uncommitted at handoff: this file,
`docs/NEXT_SESSION_PROMPT.md` and `scripts/diag/`. The user had not yet
said whether to commit them.

Not in git: `data/` and `runs/` (probabilities, packages, ZIPs, pretrain
checkpoints).

Grok's review: the user mentioned one, but it was never found on the
remote. Ask the user where it is.
