# Handoff: Track 2 state (20 October 2026, evening)

Read this, then `AGENTS.md`, `OFFICIAL_CONTRACT.md`, `docs/STRATEGY_REVIEW.md`
(why we are where we are, what is open) and `docs/SEALED_PLAN.md`. Round
details: `docs/OPTIMIZATION*.md` (R1–R8; R8 = this session). Goal: **top three in the sealed
phase**; the warm-up board is not a goal (its 1.00 entries trained on the
public warm-up test participants P04/P09).

## Dates, quotas, rules (recheck live in Codabench before any upload)

| Item | Value |
|---|---|
| Warm-up closes | 1 Nov 2026, 23:59 CET (extended after the Codabench outage; 5/day, 100 total) |
| Sealed phase | 5 Nov 00:01 → 21 Nov 23:59 CET; at most 5 uploads, 1/day |
| Ranking | best sealed score retained (`Force_Best`); scores visible, logs hidden |
| Sealed output | hidden (no logs): only upload worker-verified packages |
| Evaluation | ≤ 60 min on one A100, `submission.py` + data files; we keep ≤ 30 min |
| Audit (top 3) | training re-run from committed config, ±2σ |

The user uploads; Claude never uploads. Scherer 2015 (CC BY-NC-ND) for a
cash prize is unanswered upstream (issues #41, #52).

## Task in one paragraph

Three mental commands (motor imagery, mental subtraction, word generation)
from 4 s, 41-channel, 120 Hz EEG windows (PROTEUS, 10 public participants).
Metric: balanced accuracy per participant × session × interface cell,
averaged. Sealed = later sessions of the 10 public participants + 10 unseen.
`predict` sees one window at a time per session, in order; label-free causal
adaptation (statistics or gradient updates) within a session is allowed.

## Current candidates (`runs/upload/`, all replayed, files unchanged)

| ZIP | What | dl | ds | al | ss |
|---|---|---:|---:|---:|---:|
| `sealed_candidate_ensemble_all_r6nf_noscherer.zip` (`1d347187…`) **primary** | v2 (0.2) + 5 aligned + 5 unaligned ShallowConvNets + 5 aligned + 5 unaligned FBCNets (0.04 each), PROTEUS only | 67.2 | 80.5 | 64.9 | 71.6 |
| `sealed_candidate_ensemble_all_r6f_scherer.zip` (`accc74df…`) | same, unaligned SCN + aligned FBCNet trained with Scherer `~n~w0.1` | 67.8 | 80.5 | 66.0 | 70.8 |
| `warmup7_ensemble_official_r6f.zip` (`96cfc098…`) | r6 on the official train split (honest warm-up) | – | – | – | – |

Both sealed packages: causal session alignment (k 5), windowed class
rebalancing (last 30 windows, α 1, k 5), plain numpy/torch on CPU, shared
FBCNet filter bank: 35 ms/window Mac, 62–64 ms on a loaded A100 node
(≈ 21 min for ~20k windows). One-command rebuild of r6nf:
`scripts/build_sealed.py scripts/sealed_r6nf.json OUT`
(`evidence/sealed-rebuild-2026-10-20`; 2,654/2,655 windows agree). A
rebuild of r6f would also need a scripted Scherer download.

**Pending user actions** (warm-up closes 25 Oct 02:00 CEST): upload
`warmup7…` (expected 0.5243) and the primary sealed ZIP (expected ≈ 0.9973,
inflated by leakage; read the worker duration); optionally
`warmup8_r6nf_cbramod_gpu.zip` (`a7a4f313…`, expected ≈ 0.459; checks the
CBraMod CUDA path on the worker: duration ≲ 2 min = GPU, ≳ 4 min = CPU
fallback). Record IDs/scores/durations in `docs/COMPETITION.md`.

## Running experiments at handoff

Overnight 20 → 21 Oct on Galvani (details and licence notes in
`OPTIMIZATION_R8.md`): **2984802/2984803** FBCNet variants (stride 2/8,
16/64 spatial filters; aligned/unaligned; seeds 0–2; dl, ds, all_loso),
**2984808/2984809** REVE fine-tuned (`reve_ea`, `reve_ea_lld`; seeds 0–2;
dl+ds / all_loso), **2984618** LaBraM own recipe (dl, ds). Fetch with
`rsync -az --include='fbcnet_*' --include='reve_*' --include='labram_*'
--exclude='*' mlcloud-galvani:<ws>/runs/experiments/probs/ runs/experiments/probs_cloud_r8/`
and add `probs_cloud_r8` to `--dirs`.

Select on **`dev_loso_all`** (new, see R8) and `dev_session`, report
`dev_loso`; then evaluate on `all_loso` (P04/P09) and `seen_session` (the
latter needs separate runs for anything selected). CBraMod is not adopted
(all_loso +0.1, P04/P09 −1.8/−1.5, seen_session −1.4).

## Protocols and rules of evidence (`experiments/protocol.py`)

- Select only on `dev_loso` (8 non-test participants) and `dev_session`
  (P10 sessions 2–3, 4 cells, noisy). Evaluate on `all_loso` and
  `seen_session` only after the choice is fixed; never tune on them or on
  `official`. Both overlap the selection participants: report P04/P09 from
  `all_loso` separately (`mix_probs.py` does).
- About ±1 point is noise; prefer plateaus (seeds, weights) over grid winners.
- Every package: 100% window agreement with the harness and an official
  ingestion + scoring replay (`scripts/replay_package.py`) with unchanged
  files before staging. Native CPU replay is not proof of worker
  compatibility; time with `scripts/time_package.py` on a Galvani A100 node.

## How to run things

```bash
.venv/bin/python experiments/run_cv.py CONFIG --protocols dev_loso,dev_session [--seed N]
.venv/bin/python scripts/mix_probs.py dev_loso,dev_session v2 shallow_ea:0-4 'shallow_raw@scherer2015~n~w0.1:0-4' 'fbcnet_ea@scherer2015~n~w0.1:0-4' fbcnet_raw:0-4 [NEW_MEMBER:0-2] --dirs probs,probs_cloud,probs_r5,probs_r7,probs_fm
.venv/bin/python scripts/time_package.py runs/submissions/PKG [PKG2]
bash -c "source scripts/environment.sh && python scripts/replay_package.py PKG OUT cpu"
```

- The r6 mix above reproduces 67.81 / 80.49 / 65.97 / 70.82; r6n replaces the
  two Scherer members with `shallow_raw:0-4 fbcnet_ea:0-4`. In zsh, pass
  member lists as separate quoted words (no unquoted string variables).
- Galvani: `ssh mlcloud-galvani`, workspace
  `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026`; sync with
  `rsync -az --exclude __pycache__ experiments scripts submissions mlcloud-galvani:<ws>/`.
  Jobs: `scripts/cloud/sweep.sbatch "cfg1 cfg2" "protocols"` with
  `--array=0-2` (index = seed); `replay.sbatch`, `build_sealed.sbatch`.
  Probabilities land in `<ws>/runs/experiments/probs/`.
- Pitfalls: SSH needs the user's 2FA when the control master expires (open a
  terminal tab running `ssh -fN mlcloud-galvani` and ask the user). The
  user's other project fills the per-user job limit (jobs can wait hours);
  MPS is the fallback (FBCNet ≈ 7 min, CBraMod ≈ 20 min per dl seed). The
  Mac is sometimes loaded by other processes: don't trust local timings then.

## Exhausted (do not repeat without a new angle; numbers in R3–R7)

Alignment variants (k, stretch, train-causal); augmentation; more seeds;
v2 regularisation and member-weight tuning; learned combiner; geometric
mean; Sinkhorn balancing; AdaBN (redundant with rebalancing); self-training
on all pseudo-labels; participant identification/routing; frozen REVE;
ERP/xDAWN covariances; Dreyer 2023 (MI-only sessions break alignment);
Scherer for v2, aligned SCN or unaligned FBCNet; IFNet, TIDNet, EEGNeX,
CTNet, Conformer, ATCNet, Deep4 (aligned and unaligned).

## Next steps (detail and rationale: `STRATEGY_REVIEW.md` §7, `OPTIMIZATION_R8.md`)

1. Collect 2984514/2984515/2984618 and the paired seed-5 run. If CBraMod's
   all_loso deficit is confirmed, explain it before any further FM work
   (training-set composition, steps per epoch, normalisation); adopt an FM
   member only if all_loso and seen_session do not drop.
2. LaBraM with its own recipe (2984618); REVE fine-tuning only on Galvani
   (69 M parameters, licence "other": read it first); EEGPT (downloaded,
   BSD-3) is ~10× CBraMod's compute.
3. Re-run `build_sealed` for r6nf from a clean commit (after the user
   approves the commit) and record source hashes; `submission.py` now has the
   CBra class, so check that r6nf predictions are unchanged.
4. Finalise `SEALED_PLAN.md` (r6nf primary unless a confirmed improvement
   lands; state the audit tolerance).

Closed in R8 (with oracle bounds or negative results, see R8): trial-order
prior, label-shift-aware reference, per-session member weighting, selective
prototype TTA (T3A), task-period crops, CBraMod recipe variants.

## Repository state

`main` pushed through the round-7 docs; this handoff, `STRATEGY_REVIEW.md`,
`NEXT_SESSION_PROMPT.md` and the review fixes (CBraMod pin/strict load, net
cache, `build_sealed` train check, `replay.sbatch` slash) are in the latest
commit. Not in Git: `data/`, `runs/` (packages, ZIPs, probabilities).
Legacy scripts that document rejected experiments are kept because docs
link to them (`extract_reve.py`, `selftrain_probs.py`, `diagnose_slow.py`,
`combine_probs.py`, `external_probs.py`, `rebalance_probs.py`).
