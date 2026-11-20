# Lane T4 · independent validation of team C's session-gradient TTT (registered 20 Nov, before any score)

Question: is the TTT gain (team C dev, P07/P09 partial: A1 73.0/68.8 -> A2h 81.5/77.3) real, or dev overfitting?
Instruments never touched by team C's tuning: (1) nine Scherer 2015 people x 2 visits (22/41 channels interpolated, sign/veto
instrument; positive forecasts often flipped here, `review_19nov/forecast_calibration.md`); (2) P04/P10 returning.

## Design (code `cleanslate/ttt_valid/`, team C code `cleanslate/teamC/` imported unchanged at git HEAD 819765ac)
- Base: team C's E6 (shallow/eegnet/fbnet x seeds 0,1, 60 epochs, rot 0, `train.py` defaults) trained on ALL ten public people
  (`pkg_all`; no Scherer windows). Returning base `pkg_ret`: all ten people, P04/P10 sessions 2,3 excluded (enrolled from session 1).
- Streaming: the real `TTTModel.predict`, one window per call, `deepcopy`+`reset_state` per visit/session, full-session shuffle,
  8 orders (seeds 71000-71007; Scherer = `niv/orders.py` recipe, returning = harness `delivery`), RTX 2080 Ti / A100.
- Variants (teamC `VARIANTS`): A0 source (causal EA only), A1 + prefix BN (TTT off, the reference), A2 default TTT (4 steps, lr 1e-4),
  A2h heavy (8 steps, lr 3e-4), A3a heaviest (16 steps). Score = official cell-mean BA (`group`: person|session|interface); person
  value = mean of its cells; paired deltas vs A1, bootstrap over people AND orders (10k), 80/95 % intervals (`analyse.py`).
- Dev weighting (0.57 new / 0.43 returning; 0.2 P04 + 0.8 P10) is an UNVERIFIED assumption; not applied, people equal.

## Predictions (registered now)
Scherer (9 people, 8 orders): A1 level 40 [34, 46]; A2 - A1 +0.8 [-1.5, +3.0]; A2h - A1 +1.0 [-2.0, +4.0]; A3a - A1 +0.8 [-3, +4].
(Dev gain +1.5-8 pp shrinks on the interpolated external montage; calibration table: external magnitudes overestimated.)
Returning P04/P10 sessions 2,3 (2 people, 8 orders): A2h - A1 +2.0 [-1, +5]; A2 - A1 +1.0 [-1.5, +3.5].
Stability: collapse incidents (TTT-induced one-class run: any 40-window block with >= 85 % one predicted class while A1 on the
same stream has <= 70 %) 0 of 144 Scherer stream-orders for A2, <= 3 for A2h/A3a; no person worse than -3 pp for A2.
Rules (same reading as C1): Scherer gain SUPPORTED if the 80 % lower bound > 0 for A2h - A1 (and A2 - A1 reported alongside),
REFUTED if the 80 % upper bound < 0, else UNDECIDED (bars are reading aids; noise about +-0.5). Returning is reported
as two people, never decides. Instability finding: any TTT-induced collapse incident, or a person worse than -3 pp on Scherer,
is recorded as a veto flag against sealed use of that form regardless of the mean.

## Results
Jobs (Galvani, release `ni26_tttvalid_20261120`): train 3140385 (pkg_all, pkg_ret), Scherer 3140386 (9 tasks), returning 3140387 (2 tasks);
smoke 3140336 passed. Raw: `laneT4_scherer.json`, `laneT4_ret.json`. Orders 71000-71007, official cell-mean BA, deltas vs A1 (TTT off, prefix BN).
ms/window below are contended (several jobs per 2080 Ti), not timing evidence.

**Scherer (9 people x 2 visits, 8 orders)**: levels A0 33.7, A1 43.6, A2 43.4, A2h 40.5, A3a 40.8.
- A2 - A1 -0.15 [80 % -0.93, +0.62] (95 % -1.39, +1.02): UNDECIDED (predicted +0.8; inside its interval).
- A2h - A1 -3.03 [-4.66, -1.48]: REFUTED by the pre-registered rule (predicted +1.0 [-2, +4]; outside). A3a - A1 -2.77 [-4.74, -0.90].
- Worst person: A2 H -2.8; A2h D -9.5 (also E -6.5, A -6.7); A3a D -11.1. Veto flag (person < -3 pp) raised for A2h and A3a, not A2.
- A0 (source BN statistics) is a one-class predictor on all 144 Scherer stream-orders (33.7 %): prefix BN refresh is what makes A1 work.

**Returning P04/P10 (sessions 2,3 streamed, enrolled from session 1; TWO people, intervals not meaningful)**: A1 65.9, A2 70.4, A2h 66.3, A3a 64.4.
- A2 - A1 +4.49 (P04 +4.51, P10 +4.47; both positive in every order summary; predicted +1.0); A2h - A1 +0.42 (P04 -2.47, P10 +3.31);
  A3a - A1 -1.54 (P04 -4.78, P10 +1.70). Heavier forms hurt P04.

**Stability**: TTT-induced collapse incidents (rule above) 0 for A2, A2h, A3a on both instruments (Scherer 0/144 each; returning 0/32 each);
no one-class run >= 95 % under TTT. The failure mode is drift, not collapse: heavy forms lose accuracy over the prefix
(Scherer mean accuracy by delivered quarter, A1 .396/.445/.445/.458 vs A2h .385/.421/.404/.411; returning A2h at the last quarter
.656 vs A1 .665, worst stream P04|2 order 2 A2h .298 vs A1 .489 in the last quarter).

**Reading**: the default form (A2, 4 steps, lr 1e-4) is neutral on the interpolated external montage and +4.5 pp on the two returning
people; the heavy forms that look best on team C's dev people (A2h) lose 3 pp on Scherer (all intervals exclude 0) and are not supported
as a sealed choice. Scherer stays a sign/veto instrument (22/41 channels interpolated, base 43.6 %); the heavy-form dev gain is
not confirmed outside the ten people it was tuned on. Not covered: whether A2's returning gain survives other seeds (one base), the
new-people dev gain itself (team C's own run), and A2h on sealed-like real new people.
