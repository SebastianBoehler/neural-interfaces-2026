# Prompt for the next session

Continue our Track 2 (BCI decoding) entry for the Neural Interfaces 2026
sealed phase in
/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026. The goal
is to **rank confidently in the top three of the sealed phase**: the prize,
and above all the sponsors' internship and career opportunities, matter a
lot to me. The sealed phase opens on 5 Nov: five uploads, one per day,
best score counts, deadline 21 Nov.

## 1. Understand the current state (first)

Read in this order:
1. docs/HANDOFF.md
2. AGENTS.md and OFFICIAL_CONTRACT.md
3. docs/SEALED_PLAN.md
4. docs/OPTIMIZATION_R10.md (end: R19, round 2, R19e, round 3)
5. docs/GROUND_UP_REVIEW.md

Then look at the code: experiments/ (deep.py, models.py, features.py,
session_em.py, ensemble.py, run_cv.py), submissions/ensemble/submission.py
and scripts/build_sealed.py.

Confirm the baseline with `scripts/diag/diag_common.py`: R15 dla should be
72.11.

Summarise for me in a few lines:
- where we stand (R19e, the sealed order);
- the established bottleneck (participant-specific class directions);
- what round 3 proved: label-free teachers carry the ensemble's
  participant-structured errors.

## 2. Research block: at least 30 minutes wall-clock

Do a genuine state-of-the-art and interdisciplinary research block before
proposing experiments. Use workflows and parallel research agents (ultracode
is fine). Look explicitly beyond what docs/GROUND_UP_REVIEW.md §3 already
covered.

**Questions to answer:**
- What gives cross-subject generalisation of class directions with about
  10 training subjects?
- What creates a label source whose errors are independent of the
  ensemble?
- What do other fields do with few-domain transfer? Speech/speaker
  adaptation, medical imaging site shift, recommender cold start,
  psychometrics and item-response models, causal representation learning,
  meta-learning, test-time training with auxiliary self-supervised tasks,
  Bayesian hierarchical models.
- What is the neuroscience of MI vs mental subtraction vs word generation
  that our models may not exploit?
- What data could legally add same-task participants?
- What did 2024–26 EEG/BCI challenge winners do?

Be rigorous: for every claimed gain, ask what baseline it beat and whether
the mechanism survives our setting (causal streaming, no metadata, about
190 windows per session, a rebalanced heterogeneous ensemble).

## 3. Then iterate towards the next breakthrough

Map the remaining solution space against what is closed (docs/HANDOFF.md,
"Closed levers"; do not re-propose those without new evidence). Rank the
candidates by expected sealed gain × probability. Then run them:
- **Oracle-first gates.** Bound each idea with a true-label or ideal-case
  oracle before building the label-free version.
- **Pre-registered checks.** Use six seeds pooled and seed halves, paired
  per-participant signs and nulls. Template: scripts/diag/verify_r19e.py.
- **Compute.** Use Galvani generously for training. Use the Mac only for
  capped CPU analysis.

Aim for real jumps (≥ +1 on dla over R19e, ds not worse) rather than
knob-turning. Good candidates go through the full packaging path:
committed config, build, harness agreement 1.0, worker replica, and an
optional warm-up crash test before 1 Nov 23:59 CET if authorised. They can
then replace the weak day-2 slot (R10) or improve on R19e.

## Rules

**Evidence:**
- Select only on mean(dev_session, dev_loso_all) and report dev_loso.
- Never tune on official, all_loso, seen_session, P04/P09 or the warm-up
  board.
- About ±0.7 is seed noise; adopt at ≥ +0.5 pooled over six seeds with
  both seed halves positive.

**Packaging:**
- Every package needs 100% harness agreement, a replay with unchanged
  files and a worker-replica run before staging.
- Keep sealed runtime under about 30 min.
- Commit the config and ZIP hash before any sealed upload (top-3 audit).

**Uploads and commits:**
- Never upload to Codabench without my explicit authorisation, and check
  the live quota and phase first (Claude in Chrome).
- Ask before committing unless I've approved it.
- Never commit datasets, environments, credentials or checkpoints.
- Keep the 2026-competition submodule pinned and unchanged.

**Compute:**
- Local Python only via `NI26_MAX_GB=6 scripts/run_capped.sh LOG
  .venv/bin/python SCRIPT`, one job at a time.
- Training on Galvani. For 2FA, open a terminal tab with
  `ssh -fN mlcloud-galvani` and ask me for the code.

**Scope:**
- Skip Scherer 2015 unless I re-open it.
- No teammates.
- Keep source files below about 300 lines and diffs surgical.
