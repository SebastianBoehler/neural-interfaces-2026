# Archive: historical documents

Moved here on 11 Nov 2026. These files record how the project got to its current state. They are **not current**:
numbers, plans and "closed" verdicts in them refer to earlier pipelines and may contradict docs/STATE_11NOV.md, which
wins. Read them only to look up how something was tested.

| File | Period | What it is |
|---|---|---|
| OPTIMIZATION.md, OPTIMIZATION_R3–R10.md | 18 Oct – 1 Nov | Optimisation rounds 1–10 (tangent space, FBCNet, rebalancing, external data, teacher) |
| STRATEGY_REVIEW.md, STRATEGY_R16.md, GROUND_UP_REVIEW.md | 24 Oct – 2 Nov | Strategy reviews and solution-space maps of that time |
| HANDOFF.md, NEXT_SESSION_PROMPT.md, SEALED_PLAN.md | 30 Oct – 3 Nov | Handoffs and the first sealed-phase plan |
| REVIEW_2NOV.md, OVERNIGHT_2NOV.md, WAVE10_3NOV.md | 2–3 Nov | Reviews and overnight waves before the sealed phase |
| STATE_7NOV.md | 7 Nov | Previous entry point; its pipeline description is now in STATE_11NOV.md §3 |
| PROTEUS_SOURCE_10NOV.md, PROTEUS_ACQUISITION_10NOV.md, PROTEUS_ALPHA_CONFIRMATION_10NOV.md, PROTEUS_CHANNEL_POWER_BLEND.md | 8–10 Nov | Codex's source-screen campaign (auxiliary heads, +0.4–0.8 on a weaker base, none promoted) |
| EXPERIMENTS.md, SETUP_CHECK.md | Oct | Early procedure notes |

## Assumptions in these files that are now known to be wrong or outdated

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
