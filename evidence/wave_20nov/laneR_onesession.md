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

## Step 1 results (19 Nov ~22:40 CET; `laneR/stock_returning.py`, `laneR/oracle.py`, jsons in this dir; no new Slurm job, stored tapes only)
cap250 absolute returning BA (equal cell; 8 orders):
- P04 visit 2 / 3: 89.9 / 89.2; P10 visit 2 / 3: 94.6 / 97.2 (Lane A conditional instrument, enrolment = visit 1 only, members trained with visit 1 and without visits 2,3, verified in friday/ea_returning/returning_inputs.py). Same cells as held-new (cohort A full0): P04 ~48, P10 ~83; so one enrolled visit is worth ~+40 / +12 pp here and these two people are near ceiling.
- Recognition fires in 8/8 orders for every cell: first recognition age 10 in 18/32 streams, 28-60 in the rest (P04|2 ages 10,10,20,32,60x4; P10|2 10,10,10,28,59,60x3), routed 75-95 % of windows.
- Scherer B-ret (visit 2, 9 people, 8 orders each): ret 56.8 (full) / 58.3 (within) vs the same package without enrolment (B-new visit 2) 47.9 / 49.2; recognition 100 %, age 10, routed 91 %. Caveat (NIV.md): the same package also recognises 144/144 streams of OTHER people, so this +9 pp mixes true recognition with a universal route; Scherer is 22/41 interpolated.
Oracle (trained on the enrolment visit only, non-causal, fixed settings, test visit never used): PROTEUS O1 52.3/39.1/39.4/47.5 (mean 44.6), O2 mean 40.7; Scherer O1 51.9, O2 46.8 (9 people). Post-hoc O3 (time-domain LR, not registered) 31-44 %, chance level.
Verdict against the registered rule: headroom = max(O1,O2) - cap250 is -48 pp (PROTEUS) and -4.9 pp (Scherer, full): negative on every cell, so the 2 pp trigger does NOT fire. This is a statement about the oracle, not about cap250: simple per-person tangent models are far weaker than the stack (a person-only model trained on 100-120 windows is not an upper bound). The headroom for the eight one-session people is therefore UNMEASURED (undecided, not closed): P04/P10 are near ceiling, Scherer is a different-montage proxy, and no later labelled session of the eight exists.
Prediction check: P04/P10 headroom <= 1 pp (held in the sense that the stock is at 90-97); Scherer O1/O2 50-62 predicted, 52/47 observed (low side).
Step 2 not started (gate). Already on disk for the levers: Lane A dose (head share x2/x4/x8/x16 and hard) is monotonically harmful on the P04/P10 instrument (equal-cell -1.7/-4.2/-7.3/-9.8, hard -12.4), so more personal head weight is refuted in that form; PFT (nets fine-tuned on the enrolment visit, 11 Nov, +1.4-2.0 seen, chain-partial, P04's) is NOT shipped in cap250 (own-space is); a PFT form could only be scored on a low-accuracy returning cell, which only Scherer provides.
