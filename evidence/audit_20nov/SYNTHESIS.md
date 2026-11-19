# Audit synthesis · night of 19–20 Nov 2026

Inputs: `assumptions_rules_opus.md`, `bugs_datapath_codex.md`, `assumptions_model_codex.md`,
`first_principles_fable.md`, `sealed_composition.md`, `same_task_people_codex.md`, plus the 19 Nov lanes
(`evidence/wave_19nov/`) and `evidence/review_19nov/forecast_calibration.md`.

## What was wrong, what was not

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Hidden bug in data, labels, metric | **Refuted** for cap250. All 2,655 public windows bit-equal to the official cache; labels match BIDS cues; scorer equals the official cell mean. Three real bugs in side paths (Shin2018 pooled loader mislabels all rows; Scherer export drops run ids; legacy scorer inflates BA on predicted-only classes), none in the sealed package. | bugs_datapath |
| Misread rules / self-imposed limits | **Partly.** Pinned rules were stale: 3,600 s on one RTX PRO 4500, best score counts, Failed runs free, deadline Sat 23:59 CET. Our do-no-harm upload gates and 3,060 s / 90 MB caps are self-imposed. Streaming causality is real. | assumptions_rules |
| Wrong target (sealed mix) | **Partly.** 20,083 sealed windows ≈ 106 visits: ~60 new (10 people) and ~46 returning. Returning is ~87 % the eight one-session public people; P04/P10 ~13 %. Our returning dev arm (0.2 P04 + 0.8 P10) measured the wrong people. | sealed_composition, assumptions_model |
| Simpler/different design wins | **Refuted.** EA + tangent + LR 50.1 % held-new; the stack adds +23.6 pp. Session balancing / OT ≤ +0.6 pp even non-causally. | first_principles |
| Missing method / model / dataset | **Refuted for this deadline.** Ten FM variants in the mixer: −0.23 to +0.26 pp. No further public three-task cohort exists (PROTEUS campaign had 22 people; 10 released). | lanes C–C4, research_codex, same_task_people |

## Why the search stalled

1. **New people carry ~75 % of sealed error**, concentrated in five low-accuracy people (mean 68.8 %).
2. **The binding constraint is cross-person coverage.** 7→9 source people still adds +2.3 to +3.8 pp on one small net; no more same-task people exist publicly, and other-task corpora transfer weakly (pooled base +0.04 pp in the system).
3. **89.8 % of cap250 errors are shared by every member arm**, so new members, fusion and adaptation cannot repair them.
4. **The dev instrument cannot resolve the effects we chased** (±0.5 pp across people; 0.23–0.80 pp selection optimism from reusing ten people), and our forecasts used standalone/oracle numbers that overstate full-system gains ~10×.

## Plan to the deadline

1. Final upload = best expected gain that fits 3,600 s (free option): stack the measured small positives (F-C25 hard, CBraMod member) and score the stack jointly in the full system and on NI-V; qualify runtime on the RTX worker image. Upload Friday afternoon with the user's go-ahead, so a Failed run can be retried Saturday.
2. Optional if ready by Friday 13:00: Shin 2017 MI/SUB + Shin 2018 WORD (correctly mapped) for the raw FBCNet members, planning +0.15 pp.
3. After the deadline: returning instrument for one-session people (Scherer B-ret plus P04/P10 visit-1 enrolment), and cross-person coverage (augmentation, synthetic subjects, cross-task pretraining) as the only lever with a measured slope.
