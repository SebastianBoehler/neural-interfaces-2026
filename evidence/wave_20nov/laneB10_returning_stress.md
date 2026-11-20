# Lane B10 · returning-person stress set (registered Fri 20 Nov 2026 ~15:30 CET, before any B10 stream was scored)

Why: ~87 % of the returning sealed cells (~38 % of all cells) are eight people with ONE public session, each ~5 later sealed sessions
(`evidence/audit_20nov/sealed_composition.md`, medium-low confidence). P04/P10 (Lane A / R4) have two later visits each and sit at 90-97 %. No instrument
exists with many people, one enrolled session and several later sessions. NETBCI (19 people x 4 days, `laneB6_mi84.md`) is the closest staged proxy.

## (b) NETBCI cross-day stress (this card scores it)
Packages (exact zips, Leonardo `ni26_laneR4_20261120/zips`, all Scherer-free CBraMod B base, no NETBCI in any member): `cap250` (v66_cp1_cap250), `hardcbB`
(final stack), R4 arms `hardcbB_R1, R2, R5, R1a, R3, R4` (R4's builder, not touched), and controls `cap250_new`, `hardcbB_new` (same package, NO enrolment).
Per person k (19): day 1 enrolled with `niv/enrol.py` (the sealed-package way: contrast gallery + personal head, identity gate row, own-space members, tpri), then days 2, 3, 4 streamed
separately through the real Solver, one window per call, fresh model per (order, day). Orders: 4 full-session + 4 within-run shuffles (seeds 72000-72003), 8 x 3 streams per (variant, person).
Code `laneB10/` (`prep.py`, `run.py`, `run.sbatch`, `score.py`), release `/leonardo_work/AIFAC_P02_1675/ni26_laneB10_20261120`.
Stated limits (none hidden):
- NETBCI is 2-class (rest, right_hand; windows 1 s after the cue, 41 ch spherical-spline export, other lab and montage). The package needs three classes to enrol: right_hand -> task 0 (MI);
  day-1 `rest` windows are split alternately into placeholder classes 1 and 2 (no information in the split). Scored classes: MI recall = share of right_hand windows predicted 0;
  non-MI specificity = share of rest windows predicted 1 or 2; BA2 = mean of the two. Chance is 50. This is a stress test of recognition and the personal route, not an accuracy estimate for PROTEUS.
- The recognition gate sees a foreign lab (laneQ: stock gate over-latches there, ~13 % wrong-person latch); the shadow contrast gate is part of the package here, unlike laneQ.
- P04/P10 pairings: only visit 1 -> visits 2, 3 exist (Lane A fold packages hold visit 1 as the person's single session; R4 scores them on the Lane A instrument). Enrolling visit 2 or 3 needs
  folds whose members exclude the other visits, which do not exist and would need a rebuild of the whole stack; NOT built, so "all pairings" is NOT delivered (undecided, not closed).
  Reusing the visit-1 fold with a v2/v3 enrolment would leave v1 in the trunk and is not the one-session regime. R4's P04/P10 per-visit numbers are read, not rerun.

## Predictions (BA2 pp unless stated; 80 % intervals)
- cap250_new / hardcbB_new absolute: 53 [49, 58]; enrolled cap250 / hardcbB: 54 [49, 60]; enrolled minus new +1.0 [-3, +5] (personal head from a placeholder-class enrolment is weak; harm possible).
- Recognition (latched within the stream, any rec >= 0): 75 % of streams [50, 95]; median latency window 10-20; routed share of windows given recognised 80 % [50, 95]; days 3-4 not lower than day 2 by more than 5 points.
- hardcbB vs cap250 (enrolled): 0.0 [-1, +1]. R1/R2 vs hardcbB: +0.3 [-1.5, +2]; R5 +0.1 [-1, +1]; R1a +0.1 [-1, +1]; R3/R4 +0.3 [-2, +3]. FT-kept arms: more adaptation on a foreign-lab stream; sign unknown.
## Rules (fixed now)
Per arm vs hardcbB (paired per person, people-resampled 80 %, pooled over days and orders, enrolled streams): SUPPORTED if lower bound > 0; REFUTED if upper bound < 0; else UNDECIDED. For enrolled vs `_new`:
"harmful" if upper bound < 0. Bars are reading aids: with 19 people the interval is wide; per person/day deltas and the recognition table are the primary output. Sealed weight is not applied (a foreign lab is a stress, not a weight).
