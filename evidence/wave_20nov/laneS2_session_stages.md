# Lane S2 - session-level stage constants re-tuned for full-session shuffle (registered Fri 20 Nov 2026, ~19:10 CET, before any arm below was scored; smoke 3144243 passed the bitwise guards, no score read)

Question: the rebalance, SessionEM, TSEM/GEB, count and CSP constants were tuned under recording order on older systems (audit rows marked STALE). Under shuffled delivery are other constants better, and does a nested-honest re-tune beat the shipped values on the final stack (hardcb) and on ds_all (= hardcb + lite5 + FBL2 + per-session dispersion)? If so, is the change constants only (packageable without new code)?

## Instrument
Code `laneS2/chain/final_s2.py` (release `ni26_laneS2_20261120` on Galvani; copy of the lane P / R1 machinery). One real chain pass per distinct (system, rebalance, EM, TSEM) setting (mixer -> rebal -> SessionEM -> TSEM, then for ds_all the per-session dispersion correction, then the count filter and CSP tail). The stage functions are the shipped `gm` functions re-implemented with parameters; default parameters equal the shipped code (the base arms equal lane P hardcb and ds_all bitwise, asserted in every job; a move that leaves the output unchanged fails the job). Ten reused held-new people, orders full0 and within0 (the only orders with FT streams; other orders unscorable here), official cell-mean BA, paired people-resampled 80 % intervals (10,000 draws; people+orders where both orders are used). Dev only.
Frozen as in lane Z (stated, not silent): the GEB gate (stored incumbent rebalanced posteriors), FT banks (lite5 for ds_all), SessionEM/TSEM shadow and FT refit targets stay the incumbent ones, so the FT stream does not see changed constants; recognition and own-space routes are unrouted for held-new people. Not tunable here: SessionEM feature bins and event prior k=5 (precomputed GE features; changing them needs a feature rebuild), gate lo/hi, count thresholds/urn prior.

## Grid (wave 1; one constant moved at a time, two levels, per system and order; shipped value in brackets)
- rebalance: alpha {1.0, 1.3} [1.15]; k pseudo-windows {2, 10} [5]; window W {15, 60} [30].
- SessionEM: exponent lam {0.3, 0.7} [0.5]; n0 {30, 120} [60]; m0 {10, 40} [20]; covariance shrink {0.5, 0.85} [0.7]; boost {0, 4} [2]; age {10, 30} [20].
- TSEM: selected fraction rho {0.5, 0.85} [0.7]; shrink {0.5, 0.85} [0.7]; m0 {10, 40} [20]; n0 {30, 120} [60]; lam {0.10, 0.22} [0.15]; boost {0, 8} [4]; age {10, 25} [15].
- tail: count filter off; CSP off; CSP refit cadence {10, 40} [20]; CSP selected share q {0.3, 0.7} [0.5]; CSP dose cap {0.05, 0.15} [0.10].
40 moves x 2 systems x 2 orders (stage moves need a chain pass, tail moves reuse the base chain).

## Selection rule (fixed now) and wave 2
Held-out person j: per constant take the better of its two levels if its mean train delta (equal-cell BA vs base of the same system, train = the other nine people) is > 0; the selected moves of a scope are applied together, moved a fraction s of the way from the shipped value to the selected one (s 1.0 and 0.5; integers rounded; on/off flags only at s 1.0). Selection data: (self) the same order's other nine, (other) the OTHER order's other nine (nothing chosen on the scored order or person), (both) both orders' other nine. Scopes: rebalance, SessionEM, TSEM, tail (count+CSP), all. Every form is one chain pass with per-person settings (the stages are per-session independent: each distinct setting is run on all rows and each person takes its own). Un-nested reference: the same rule fitted on all ten people and both orders. Per-stage honest gain = the scope-only form; combined = scope all. Reported vs hardcb and vs ds_all, each with the people 80 % and the people+orders 80 % interval, and the nested vs un-nested gap.

## Predictions (pp, both-order mean vs the system's own base; 80 % half-width about +-0.6)
1. Single moves: median |delta| 0.15, largest positive +0.4. Under full shuffle class proportions are stationary within a session, so longer memories (rebalance W 60, k 10) are the likeliest positive moves (+0.2..+0.4 on full0, about 0 on within0); alpha 1.3 and boost 0 negative; count off about 0.0 on full0 and about -0.5 on within0 (lane Z); CSP off about +0.1 full0, -0.5 within0.
2. Un-nested optimum (about 12 constants chosen on ten people): +0.6 [0.0,+1.2] vs base; nested-honest (all, s 1.0): +0.1 [-0.5,+0.7]; s 0.5: +0.2 [-0.3,+0.7]; per-stage honest gains each within -0.2..+0.3.
3. ds_all and hardcb agree in sign for every stage-level conclusion (the correction acts after TSEM).
Rule: a form is "positive" if the both-order nested-honest gain vs the system base >= +0.3 with the people+orders 80 % lower bound > 0 (then the exact constants are given as a package spec); inside +-0.5 of base "undecided" (AGENTS.md); nothing is closed on P10/P04 alone, both reported.

## Run log
(appended below as jobs run)

Smoke 3144243 (bases, bitwise vs lane P: hardcb and ds_all) and 3144361 (one move per constant, per-person form) passed; smoke 3144453 reruns with the no-op guard on stage moves. Wave 1 (grid, 4 parts x 2 orders): Galvani 3144454-3144461 (afterok smoke 3144453), scorer afterok (see below). Chain passes are about 10 s each; the queue is the shared account's 36-job limit.

## Wave 1 result (Fri 20 Nov ~20:50 CET; jobs 3144454-61 (grid, 4 parts x 2 orders), scorer 3144462; smoke 3144453 passed all guards, only the receipt write collided with the earlier smoke file; cells `laneS2/data/cells_w1.json`, table `laneS2/data/report_w1.txt`)
Base arms equal lane P hardcb and ds_all bitwise in every job; every move changed the output. Single moves vs the system's own base (both-order mean; people 80 %): the shipped constants are close to a local optimum.
- ds_all: only csp.q=0.7 is positive, +0.21 [+0.07,+0.36] (full0 +0.31, within0 +0.11; people+orders [-0.00,+0.44]); csp.cap 0.05 +0.04, csp.k 40 0.00, tsem.nmin 10 0.00. Every rebalance, SessionEM and TSEM move is negative (-0.05..-0.67), most with an upper bound near 0; count off -0.35, CSP off -0.25.
- hardcb: tsem.bts=8 +0.19 [+0.06,+0.32] (full0 +0.11, within0 +0.27), tsem.n0=30 +0.15 [-0.03,+0.33] (+0.29/0.00), tsem.m0=10 +0.08, tsem.lam=0.22 +0.07, rebal.alpha=1.0 +0.02, csp.k=10 +0.02; all other moves <= 0 (rebal W 60 -0.29, rebal k 2 -0.43, em.lam .7 -0.37, count off -0.26, CSP off -0.50).
- Prediction 1 (longer rebalance memory positive under full shuffle) refuted: W=60 is -0.20 on full0, k=10 -0.15, alpha 1.3 -0.08; nothing reached +0.4. Count off is 0.00 (hardcb) / +0.03 (ds_all) on full0 and -0.53/-0.64 on within0, as in lane Z. hardcb and ds_all disagree on which TSEM moves help (bts 8: hardcb +0.19, ds_all -0.15), i.e. the winners are system-specific and inside noise.
Wave 2 (nested + un-nested forms, 80 arms per order, selection rule as registered): 3144564-79, scorer 3144580 (afterok).
