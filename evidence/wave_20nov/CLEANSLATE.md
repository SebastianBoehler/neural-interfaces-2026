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

## Harness usage (`cleanslate/harness/`, built 19 Nov night)

Release on Galvani: `/mnt/lustre/work/gehler/gfh527/ni26_cleanslate_harness` (rsync `cleanslate/` into it:
`rsync -a --delete --exclude __pycache__ cleanslate mlcloud-galvani:$REL/`). Nothing runs on the login node except `sbatch`.

**Team contract.** `cleanslate/<team>/train.py` uses `harness/data.py` (`train_parser()`, `load_train(people, first_session_only)`)
and writes `--out DIR/submission.py` (`class Solver(CompetSolver)`, official interface, no other `.py`) plus weights. It must use only the
windows of `--people`, and only session 1 of `--first-session-only` people. Env switches (ablations) are read in `train.py` and
written into the package config; the Solver must read the config, not the environment. Env values may not contain commas.

**Run a team** (on the Galvani login node, stdlib python):
```
cd $REL; CS_REL=$REL python3 cleanslate/harness/run_lopo.py --team T --orders 71000,71001 [--people P01,P04] [--tag X] [--env A=1] [--time 06:00:00] [--partition a100-galvani --gres gpu:a100:1]
```
- held-new: per person P (10 jobs) `train.py --people <other nine>`, stream all sessions of P; returning: P04 and P10 (2 jobs) `train.py
  --people <all ten> --first-session-only P`, stream sessions 2,3. One array task per fold = train then stream (GPU), then a dependent
  CPU job `score.py`. Output `runs/T/<tag>/{cells.npz,scores.json,predictions.npz}` (per-window labels by data row, per order and fold).
- Streaming = official evaluator semantics: `fresh_model` (deepcopy + `reset_state`) per session, one (1,41,480) window per
  `predict`, int64 label (1,) checked, full-session shuffle by seed (`rng [seed,0,0]`, same recipe as `niv/orders.py` "full"; the seed
  streams for 71000-71007 are shared with NI-V but cut different sessions). Score = official `group_scores(balanced_accuracy)` over
  participant/session/interface cells, from the submodule's `benchmark_utils.metrics`.
- Failed fold means no `scores.json` (score job is `afterok`; `score.py` also checks every fold's prediction file).

**Compare**: `python cleanslate/harness/compare.py RUN_A RUN_B` (numpy only, any machine with both `cells.npz`): A minus B in pp,
person value = mean of the person's cells, paired bootstrap over people and orders (10k draws), 80 % and 95 % intervals,
held-new (10 people), returning (**P04/P10 only, two people**, later sessions) and the sealed-weighted 0.57 new / 0.43 returning. The
returning weight rests on two people, so its interval is wide; the 0.2 P04 + 0.8 P10 weighting is not applied (equal people).
If the runs share no orders (cap250 reference) each side is averaged over its own orders first and the output is flagged.

**cap250 reference** (`harness/reference_cap250.py`, `ref_cap250.sbatch`; run `runs/cap250/c25d_full0`): NO cap250 LOPO fold packages
exist (held-new numbers are replays of the nine-person frozen banks plus the stage chain, not Solver ZIPs). The reference reads the
`stock|terminal` score vectors of the c25d full0 bank (`ni26_friday_c25d_held_v1_20261119/full0.npz`; shared fresh seed, legacy
order id 0, not 71000-71007), argmax, and scores them with the same `score.py`. **Held-new only, one order, no returning arm,
not re-streamed.** Score: 73.739 % cell mean over 28 cells (matches the 73.7 held-new figure in the programme text).

**Example team** (`cleanslate/example/`): causal EA + log-Euclidean tangent space + balanced LR in a Solver; a smoke system (about 38-40 %).

**Clean reference system** (`cleanslate/reference/`, for idea screening; every component can be switched off): causal EA of the signal
(running mean covariance, identity shrinkage for 10 windows) + compact EEGNet (raw, 41 ch, trained on the nine people, 40 epochs
AdamW/OneCycle, balanced CE) + causal adaptive BatchNorm (eval statistics = (20 train + n prefix) blend over the delivered prefix)
+ optional class-balance prior (logit minus 0.5 log running mean of the session's softmax). Switches (read by `train.py`, stored in
`config.json`): `CS_REF_EA`, `CS_REF_ADABN` (default 1), `CS_REF_PRIOR` (default 0).
**Ablate**: `python cleanslate/harness/ablate.py submit --team reference --off CS_REF_EA=0 --orders 71000,71001` retrains every fold with
that component off (same rosters, orders) into tag `abl_CS_REF_EA0_o...`; `python cleanslate/harness/ablate.py report RUN_FULL RUN_ABL`
prints full minus ablated (positive = the component helps) with the same people+orders intervals. The component is always measured
inside the whole system. Other teams expose their own switches the same way.

**Smoke and reference numbers (19 Nov, orders 71000/71001, 12 folds each, RTX 2080 Ti; runs under `$REL/runs/`)**: cap250 c25d-full0 reference
73.74 % held-new (28 cells). Clean reference 44.26 / 45.47 % held-new, returning 45.80 / 44.73 %; example 41.92 / 42.85 held-new, 44.05 / 46.21
returning. Reference minus example: held-new +4.15 pp [80 % +2.21, +6.11], returning +0.13 [-2.18, +2.44] (P04/P10 only), sealed-weighted
+2.43 [+0.85, +4.01]. Ablation path checked (EA off on P01: 49.4 vs 56.5 %, `config.json` shows `ea 0`). Reference is deliberately plain
(about 30 pp below cap250 held-new): it screens components, it is not a competitor. Operational: `galvani-cn122` gives CUDA_ERROR_UNKNOWN
(excluded by default, `--exclude`); `run_lopo.py --tag T --resume` re-runs only folds without predictions.
