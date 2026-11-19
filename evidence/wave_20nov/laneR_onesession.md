# Lane R · one-session returning (registered Thu 19 Nov 2026 ~21:50 CET, before any oracle was scored)

Why: ~87 % of sealed returning cells (~37 % of all cells) are the eight public people with ONE public session; P04/P10 tuning
does not represent them (evidence/audit_20nov/sealed_composition.md). Nobody measured "one labelled session known -> new session".

## Step 1 instrument (what exists, what is new)
- (a) P04/P10 enrolled on visit 1 only, streamed on visits 2 and 3, 8 orders: this is exactly Lane A's conditional instrument
  (`laneA/`, fold packages `P04_s23`/`P10_s23`, outputs `ni26_lane_a_returning_20261119/out`, stock arm = cap250 RouteB).
  I do NOT rerun it; I read absolute BA, recognition age and routed windows from the stored tapes (labels opened only in
  `laneR/stock_returning.py`). Seen BEFORE this registration (honest): stock BA per cell, recognition ages, routed counts.
  Limits: conditional replay (frozen order banks, 2 people), both interfaces of a visit streamed together.
- (b) NI-V B-ret: stored `out/ret_cap250_k*` (enrol visit 1 of Scherer k, stream visit 2; 8 full + 8 within orders), absolute BA
  and recognition from the meta rows; new-person reference = stored `out/cap250` (B-new, visit 2, same orders). Scherer has 22/41
  channels interpolated; the zero-routing arm already showed recognition fires on 144/144 other-person streams (cannot
  discriminate there), so Scherer B-ret mixes true recognition and a universal route.
- (c) Oracle within-person models, `laneR/oracle.py` (CPU, non-causal, trained on the enrolment visit's labels only, evaluated on
  the later visit; labels of the test visit are used only for scoring):
  O1 = person-only: bands 8-13/13-20/20-30 Hz (order-4 zero-phase), per-window covariance (trace ridge 1e-3), EA with the
  enrolment-visit mean covariance, tangent space at identity, standardise, LR C=0.1 balanced.
  O2 = O1 features, trained on the person's enrolment visit (weight 3) plus every other person's visits, each (person, visit)
  EA-referenced to its own mean covariance (an upper reference for "cross-person + personal").
  No tuning on test visits (C, weight, bands fixed here). Cells: PROTEUS (person, visit 2|3) and Scherer (person, visit 2),
  BA over all windows of the visit (both interfaces), equal-weight person mean.
Headroom = max(O1, O2) - cap250 per cell, equal weight per person (P04/P10 each 1/2 of the PROTEUS half; Scherer 9 people).

## Prediction (numeric, before scoring)
P04/P10: cap250 stock already 89.6 / 95.9 (seen), so headroom against O1/O2 <= 1 pp (P(headroom >= 2 pp) ~ 10 %). Scherer:
cap250 ret 56.8 (seen, full), O1 within-person trained on 105-120 windows of 22-interpolated channels ~ 50-58 %, O2 ~ 55-62 %;
headroom +0..4 pp (P(>= 2 pp) ~ 40 %), dominated by individual people. Interpretation rule: a lever hunt (Step 2) is justified only if
max(O1,O2) - cap250 >= 2 pp on Scherer AND the same sign (>= 0) on P04 and P10, or if O2 - cap250 >= 2 pp on P04/P10. Otherwise
"cap250 is near the oracle in the cases we can measure; the eight remain unmeasurable" and Step 2 is reported as undecided (the
oracle is not an upper bound; AGENTS: rank-1 is not a bound), not closed.
