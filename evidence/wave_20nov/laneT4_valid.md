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
(pending)
