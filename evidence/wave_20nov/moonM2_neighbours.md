# Moonshot M2 - person-experts weighted by unlabeled-prefix similarity ("neighbours")

Registered Thu 19 Nov 2026 ~22:12 CET, before any expert posterior was scored. Read first: STATE, audit SYNTHESIS, first_principles_fable, laneH card.

## Idea
A pooled model averages over people whose EEG differs; a new person may resemble only a few. Train per-training-person experts
(and the pooled model), weight experts per session from the delivered prefix only (no labels).

## Stage 0 - ceiling (fast CPU; EA + 5-band tangent + standardised LR C=0.01, LOPO over P01-P10, 28 cells, official cell mean)
Experts: one LR per training person (9 per fold; own sessions only). Comparator: pooled LR on the same 9 people (first-principles ref 50.1 held-new).
Non-causal EA (whole-session reference) is used for the ceiling only; causality is added in stage 1.
- Oracle-best single expert per held-out person (label-selected, optimistic) and a split-half version (select on odd windows, score on even) that removes most selection bias.
- Prediction: single-person experts score far below pooled (about 35-42 % vs 50) because 265 windows are few; oracle best expert 45 %, i.e. -5 pp vs pooled;
  oracle best expert-of-clusters / pooled + expert mixture may still help. P(oracle best single expert >= pooled + 3 pp) = 15 %.
- Stop rule (user): oracle best single expert per new person not >= +3 pp over pooled => stop early, record "single-person experts refuted at this dose" and test the
  stated variant "pooled + top-k expert mixture oracle" once before closing (the user's rule is about single experts; a mixture ceiling is a different form).
Stages 1-3 (causal similarity weights, raw FBCNet per-person fine-tunes, standalone under 8 orders, cap250 1/6 slot) run only if stage 0 passes.
Kill: Friday 06:00 CET if standalone <= +2 pp over pooled. Bars are reading aids; undecided inside +-0.5.

## Stage 0 result (Thu 19 Nov ~22:13 CET; local, `moonM2/stage0.py`, log `moonM2/stage0.log`, 20 s; EA+5-band tangent+LR C=0.01, non-causal EA, 28 cells)
Held-new, person-mean / cell-weighted (official-style) BA, 9 training people per fold:
| | person-mean | cell-weighted |
|---|---|---|
| pooled LR (matches first-principles 50.1) | 52.89 | 50.04 |
| mean single-person expert | 43.07 | 41.36 |
| **oracle best single expert per new person (label-selected, optimistic)** | 52.99 (+0.10) | 50.76 (+0.72) |
| best expert chosen on odd windows, scored on even (selection-bias-free) | 50.04 (-2.85) | 48.11 (-1.93) |
| unweighted mean of all 9 experts | 53.69 (+0.80) | 50.53 (+0.49) |
| top-3 experts (label-selected on all windows) mixed | 55.24 (+2.35) | 52.74 (+2.70) |
| top-3 chosen on one half, scored on the other | 53.69 (+0.80) | 51.16 (+1.12) |
| pooled + half-split top-3 mix | 55.08 (+2.19) | 51.90 (+1.86) |
Per person the oracle best expert beats pooled only for P03, P05, P10 (the weak ones; +4.7, +11.6, +4.8) and loses for the rest; the best expert differs by person (P02, P08, P03, P05, P01, P07), no shared neighbour.
Rule applied: oracle best single expert +0.10 / +0.72 pp vs required +3 -> **stop early; single-person experts refuted at this dose (270 windows per expert)**.
The one stated variant (mixture ceiling) with labels used for the choice reaches only +1.9 to +2.2 pp half-split and +2.4 to +2.7 optimistic, below +3; a label-free similarity weight can only
recover part of that, and this ceiling is for the small linear model (50 %), not for the stack (73.7), where the 89.8 % shared-error finding says gains shrink ~10x.
Not run (stated, not zero): fine-tuned FBCNet per-person copies, cluster experts (k-means of 2-3 people), causal similarity weights, the cap250 slot. Not covered: per-person experts at larger dose
(e.g. expert = person + nearest sessions of others), which is a different design (similarity-weighted pooled training), untested.
