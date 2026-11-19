# Clean-slate program · 20–21 Nov 2026

User, 20 Nov ~00:30 CET: get away from cap250, build an independent system that uses everything the challenge allows,
the newest models and datasets and all compute, and aim for top 3.

## Why

- Member improvements do not pass through cap250's chain: B5 (more training people) lifts one family standalone
  +4.6 pp (N 5→9) and the full system −0.4 pp; ten foundation-model variants inside the mixer: −0.23 to +0.26 pp.
- cap250's stages add ~14 pp over its best member (59.3 → 73.7 held-new), so a new system must earn that adaptation
  gain itself, not only better features.
- Facts to design against: `evidence/audit_20nov/SYNTHESIS.md`, `first_principles_fable.md`, `sealed_composition.md`,
  `assumptions_rules_opus.md`, `research_codex.md`, `same_task_people_codex.md`.

## What is allowed (official, upstream 8447e181)

- One `Solver` package; `predict` gets one (1,41,480) µV window at 120 Hz and returns one label immediately; it may use
  the current and all earlier delivered windows of the session (no labels, no metadata); full state reset per session.
- 3,600 s total on one RTX PRO 4500 (32 GB) for ~20,083 windows in ~106 sessions; worker image: torch cu130,
  braindecode, mne, moabb, sklearn, scipy; staged HF `brain-bzh/reve-base`; no compilers.
- Training is free: any public data whose licence permits, any pretrained model, any compute. Declare corpora.
- Sealed mix (inferred): ~60 sessions of 10 new people, ~46 returning sessions, ~87 % of those from the eight public
  people with one public session.

## Common contract (so systems are comparable)

Each team delivers `cleanslate/<team>/train.py --people P01,...` → a package directory with `submission.py`
(`class Solver`, official interface) and weights. The shared harness (`cleanslate/harness/`) trains LOPO folds,
streams every held session through the real `Solver.predict` under full-session shuffle (orders 71000–71007),
scores the official cell-mean BA, and also scores the returning setting (P04/P10 enrolled from their first session).
cap250 is scored by the same harness as the reference.

## Timeline

- Fri 20 Nov 12:00 CET: each team has a first whole-system dev score (at least 2 orders).
- Sat 21 Nov 08:00: best versions scored on 8 orders + returning + NI-V Scherer.
- Sat 12:00: winner packaged, worker-image timed (< 3,600 s with margin), reset/no cross-session state checked.
- Sat ~14:00–16:00: final upload of the better of {clean-slate winner, Friday stack} with the user's explicit go-ahead.
  Only one upload remains, so the Friday stack is NOT uploaded on Friday.

## Rules for teams

Pre-register each scored system in `evidence/wave_20nov/cleanslate_<team>.md`; Slurm only, never on login nodes;
smoke first; never edit `submissions/*/submission.py`; commit only your own paths; never upload.
