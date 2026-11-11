# Archive: historical documents

Moved here on 11 Nov 2026. These files record how the project got to its current state. They are **not current**:
numbers, plans and "closed" verdicts in them refer to earlier pipelines and may contradict docs/STATE.md, which
wins. Read them only to look up how something was tested.

| File | Period | What it is |
|---|---|---|
| OPTIMIZATION.md, OPTIMIZATION_R3–R10.md | 18 Oct – 1 Nov | Optimisation rounds 1–10 (tangent space, FBCNet, rebalancing, external data, teacher) |
| STRATEGY_REVIEW.md, STRATEGY_R16.md, GROUND_UP_REVIEW.md | 24 Oct – 2 Nov | Strategy reviews and solution-space maps of that time |
| HANDOFF.md, NEXT_SESSION_PROMPT.md, SEALED_PLAN.md | 30 Oct – 3 Nov | Handoffs and the first sealed-phase plan |
| REVIEW_2NOV.md, OVERNIGHT_2NOV.md, WAVE10_3NOV.md | 2–3 Nov | Reviews and overnight waves before the sealed phase |
| STATE_7NOV.md, STATE_11NOV.md | 7 Nov, 11 Nov | Previous entry points; replaced by docs/STATE.md on 11 Nov (22:30 UTC) |
| BASELINE_RESULT.md | Oct | The reproduced official linear baseline (37.75 %) |
| PROTEUS_SOURCE_10NOV.md, PROTEUS_ACQUISITION_10NOV.md, PROTEUS_ALPHA_CONFIRMATION_10NOV.md, PROTEUS_CHANNEL_POWER_BLEND.md | 8–10 Nov | Codex's source-screen campaign (auxiliary heads, +0.4–0.8 on a weaker base, none promoted) |
| EXPERIMENTS.md, SETUP_CHECK.md | Oct | Early procedure notes |
| AUDIT_M7R4.md | 4 Nov | Rebuild recipe of `cand_m7r4`, a four-family teacher package that was staged and never uploaded |

## Assumptions in these files that are now known to be wrong or outdated

- **"A stronger or different member will lift the chain."** Twelve cases say no, including calibrated ones
  (docs/LEDGER.md, "Members and features"). Members share their errors.
- **"Dev seen gains transfer like dev new gains."** Dev weights P04 and P10 equally; on the sealed set P04 probably
  has 3 of about 46 returning sessions. Every returning-person gain so far is P04's (docs/STATE.md §4).
- **"Better pseudo-labels are available with more effort."** Label accuracy for past windows stops at 77 % for every
  label-free method tried (docs/PREDICTIONS_11NOV.md §28, §31).
- **"The chain replay lives in a temp scratch path."** It is `replay/` in the repo since 11 Nov.

- **"Cue order is random; only class counts per run are structured."** Graz runs follow a weakly shuffled
  WORD → SUB → MI cycle (docs/PREDICTIONS_11NOV.md §9).
- **"5 sealed uploads" / "10 in total".** The allowance is 7 per team, 1 per UTC day.
- **"Windows are strictly chronological within a session."** The organisers removed that promise on 10 Nov; our replay
  still suggests a chronological order, but it is no longer guaranteed.
- **"The sealed set is 20 participants × 6 sessions, 57 % new people."** The organisers withdrew the numbers; the cohort
  size is private. Only the mix (later sessions of public people plus new people) is confirmed.
- **Scores before 10 Nov** (0.7795, 0.8197, 0.8240) were replaced by the rescoring (0.7723, 0.8080, 0.8135).
- **"Foundation models / more seeds / session-stage tweaks might still help."** Tested and closed for the recipes
  listed in docs/REVIEW_5NOV.md; a closed verdict covers only the method class that was tested.
- **Runtime budget "60 min, we use 45".** The shipped package was rescored at 3,732 s; the fast package replaces it.

## Code removed in the cleanup of 11 Nov 2026

Removed because nothing current referenced it any more: not the packages, their audit recipes, the replay, the
studies of 11 Nov or any current document. Everything is in git history; `git show 05d2335:<path>` recovers a file.

| What | Files |
|---|---|
| Early experiment modules | `experiments/`: calibrate, cluster_bound, identify, personal_bound, soft_route, tta_eval, ttime_eval |
| One-off scripts of October | `scripts/`: combine_probs, diagnose_slow, export_jit_member, export_mental, external_probs, merge_shin_ab, rebalance_probs, selftrain_probs, sort_probs, summarize_results, swap_external, time_package, train_tangent, validate_baseline, run_baseline.sh, fit_channel_power_blend |
| Old diagnostics | `scripts/diag/`: agent_probe, check_lam, check_sd_floor, eval_members, families, riemann_heads, stress_package, verify_r19e, weight_edit |
| Old job files | `scripts/cloud/`: external, swap_external, tta, ttime (.sbatch) |
| Package configs of superseded candidates | `scripts/sealed_r10*.json`, `r14`, `r15`, `r19e_l25`, `r8t`, `r8t4`, `r9b`, `r9c`; `warmup8/10/12*.json`; the `cand_m7r4` audit script and manifest |
| Codex's source-screen campaign (8–10 Nov, closed) | 55 files under `runs/rank1_8nov/` are no longer tracked; they are still on the Mac in that folder |

Kept on purpose although old: `scripts/sealed_r19.json`, `sealed_r19e.json`, `sealed_r9.json`, `sealed_r6nf.json` and two
warm-up configs (referenced by the upload log, the build script or the competition notes).
