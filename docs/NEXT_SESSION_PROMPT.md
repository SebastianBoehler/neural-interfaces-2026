# Prompt for the next session

Continue our Track 2 (BCI decoding) entry for the Neural Interfaces 2026
sealed phase in
/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026. The goal
is to **rank confidently in the top three of the sealed phase**: the prize,
and above all the sponsors' internship and career opportunities, matter a
lot to me. The sealed phase runs until 21 Nov 23:59 CET: five uploads in
total (three used by 7 Nov), at most one per UTC day, best score counts.

## 1. Understand the current state (first)

Read in this order:
1. docs/STATE_7NOV.md (the entry point: baseline, sealed history, board,
   runtime, what is closed and what is open)
2. AGENTS.md and OFFICIAL_CONTRACT.md
3. docs/SEALED_UPLOADS.md (upload log) and docs/AUDIT_SBN.md (audit recipes)
4. docs/REVIEW_5NOV.md (5–7 Nov evidence and closed directions)
5. runs/archive_7nov/INDEX.md (archived dev bases, preregistrations and
   confirmations of 5–7 Nov)

Older docs (HANDOFF.md, SEALED_PLAN.md, OPTIMIZATION_R10.md,
GROUND_UP_REVIEW.md) are background; read them only for a specific number.

Then look at the code: submissions/ensemble/submission.py (lean since 7 Nov),
experiments/ (ts_core.py, ts_core2.py, tsem.py, session_bn.py,
session_em.py, route_stream3geb.py) and the export scripts
(scripts/export_*.py, scripts/check_compat.py).

Summarise for me in a few lines:
- where we stand (route3geb 0.8240, 4th; uploads and runtime left);
- the established bottleneck (per-session class directions);
- what transfers to the sealed board (big steps about 1:1, small stacked
  gains about 0.6×).

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

Map the remaining solution space against what is closed (docs/STATE_7NOV.md
and docs/REVIEW_5NOV.md; older: docs/HANDOFF.md, "Closed levers"; do not
re-propose those without new evidence). Rank the candidates by expected
sealed gain × probability. Then run them:
- **Oracle-first gates.** Bound each idea with a true-label or ideal-case
  oracle before building the label-free version.
- **Pre-registered checks.** Use six seeds pooled and seed halves, paired
  per-participant signs and nulls. Template: scripts/diag/verify_r19e.py.
- **Compute.** Use Galvani generously for training. Use the Mac only for
  capped CPU analysis.

Aim for real jumps (rank 1 is 1.4 points away: about +1.4 dev J at full
transfer, +2.3 at 0.6×) rather than knob-turning. Good candidates go through
the full packaging path: committed config, build, harness agreement 1.0,
replay with unchanged files, worker replica and an audit recipe.

## Rules

**Evidence:**
- Since 5 Nov the dev scorer is the person-equal J (0.5 new + 0.5 seen;
  runs/archive_7nov/own/pe_metric.py). Choose on exploration bases with
  split-half or nested selection, then confirm on fresh bases
  (runs/archive_7nov/bases/) against a bar written down beforehand.
  The older rules below date from the warm-up.
- Select only on mean(dev_session, dev_loso_all) and report dev_loso.
- Never tune on official, all_loso, seen_session, P04/P09 or the warm-up
  board.
- About ±0.7 is seed noise; adopt at ≥ +0.5 pooled over six seeds with
  both seed halves positive.

**Packaging:**
- Every package needs 100% harness agreement, a replay with unchanged
  files and a worker-replica run before staging.
- Keep the sealed runtime under 60 min with margin: the baseline already
  takes 44.8 min, so new stages must be cheap or pay for themselves.
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
