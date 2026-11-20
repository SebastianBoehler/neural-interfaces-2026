# Lane R1 - big stack with nested re-weighted mixer (registered Fri 20 Nov 2026 ~14:15 CET, before any arm below was scored)

Hypothesis (user): the fixed mixer weights absorb component gains; a stack of every positive component with jointly re-chosen family masses passes more of them.
Lane Z owns the stage profile and the plain mixer levers (raw-slot x2/x4, learned weights on hardcb/stock); this lane does NOT repeat those on the stock system. It re-weights the ds_all system and adds the new in-mixer members.

## System and instrument
Code `laneR1/chain/final_r1.py` (copy of the lane P chain; one real chain pass per arm: mixer -> rebal -> EM -> TSEM -> dispersion (stand-in segment member, per-session history, stock-raw pseudo-labels, as lane P ds_all) -> count/CSP).
Release `ni26_laneR1_20261120` on Galvani. Base = ds_all = hardcb + lite5 + FBL2 + dispersion. Ten reused held-new people, orders full0 + within0 (the only orders with FT streams), official cell-mean BA, paired
people-resampled 80 % intervals (10,000 draws; people+orders where both orders are used). Mixer has 13 items, stock weights c1 1, hb 1, '1' .5, '2' .5, '3' 1, '4' 1, '10' 5 (teacher), '9' 1, sc_fbraw/sc_fb/sc_sinc/sc_fbl/sc_rest 1 each (sum 16).
New in-mixer members (full0 ONLY: the REVE/Pratibha/CBraMod/X banks exist for orders full0..7 and not within0; stated, not silent): reve, prat, cbx (3-seed C ensembles), xfam (B6 X, 3 seeds), each as one extra mixer item at 1 or 3 weight units.
Guards: base arm must equal lane P ds_all terminal bitwise (both orders); stock equals the release; the rebuilt linear mixer equals the real mix to 1e-12 for every arm. Failure invalidates the wave.

## Waves (fixed now)
1. Coordinate wave (per order): base; each of the 13 mixer weights x0.5 and x2 (26 arms); each extra member at 1 and 3 units (8 arms, full0). 35 arms full0, 27 within0.
2. Nested forms, chosen only from wave-1 chain scores: for held-out person j use the other nine people only. Per key take the better of its two moves if the mean train delta is > 0; combine multiplicatively with shrink s (factor^s), s in {0.5, 1.0}. Selection data: (a) the same order's other nine people, (b) the OTHER order's other nine people (nothing chosen on the scored order), (c) both orders. Extras selected on full0 only. One chain pass per (selection source, s) with per-person weights = all ten held-out people at once.
   Un-nested reference: the same rule fitted on all ten people (global weights); the gap nested vs un-nested is the real/selection split.
3. Proxy-learned forms (ridge log-likelihood pool on the tapped member posteriors of the base arm, LOPO, penalty toward stock; lambda grid) only if wave 1 shows that coordinate moves are not all inside noise. Reported as variants of wave 2.

## Predictions (pp, both-order mean vs ds_all, nested-honest, 80 % half-width +-0.6)
- Best nested form (2a/2b/2c at s=0.5): +0.10 [-0.5,+0.7]; at s=1.0: +0.00 [-0.7,+0.7]. Un-nested optimum +0.5 [0,+1.0] (selection among ~35 arms on ten people).
- Single coordinate moves: median |delta| 0.15, largest positive +0.4; extras (reve/prat/cbx/xfam): each between -0.3 and +0.3 vs ds_all in full0 (members 53-56 % BA, one-person gains P01/P06).
- Rule: a form is "positive" if nested both-order mean vs ds_all >= +0.3 with the people+orders 80 % lower bound > 0 (then the packaging spec is given); in between "undecided" (AGENTS.md); nothing closed on P10/P04.
Not covered: returning P04/P10, NI-V, recognition and own-space routes (unrouted for held-new in this replay), orders 1-7 (no FT streams), stand-in dispersion member.

## Run log
Harness fixes before any scoring (smoke 3142177 base, 3142205 mult/add/W/Wadd arms, all guards passed; mixdiff 0.0): the F-C21 `doubling` wrapper asserts a linear relation to the stock weights, so FBL2 is built into the R1 mixer (sc_fbl counted twice, asserts share 1/17 and warm_rows 2375). Rebuilt linear mixer equals the real mix exactly.
Wave 1: Galvani 3142238-3142241 (full0, 4 parts), 3142242-3142245 (within0), scorer 3142246 (afterok), `ni26_laneR1_20261120/chain/cells.json`. Analysis `laneR1/analyse_r1.py`.

## Wave 1 result (Fri 20 Nov ~13:25 CET; cells in `laneR1/data/cells_w1.json`; base = ds_all bitwise equal to lane P in both orders, stock bitwise)
Single coordinate moves vs ds_all (both orders, pp): stock weights are a local optimum. Positive: m_4 x2 +0.27 [-0.09,+0.65] (full0 +0.46, within0 +0.08), m_2 x0.5/x2 +0.08, m_1 x0.5 +0.06. All other mixer moves negative, most outside zero (teacher x2 -0.95, c1 x2 -0.87, hb x2 -0.41, sc_rest x0.5 -0.37, sc_fbl x0.5/x2 -0.34/-0.42). Extras (full0 only, vs base, 1 / 3 units): prat +0.20 / +0.07, reve +0.20 / -0.55, cbx -0.21 / -1.05, xfam -0.55 / -1.05.
Reading: the fixed weights are not an absorber for single-member mass shifts on this system; mass moved off stock lowers the score nearly everywhere. Wave 2 (nested forms) 3142518-21, wave 3 (proxy-learned, `laneR1/proxy_fit.py`: pre-chain ridge log-likelihood pool, LOPO, lam 0.003-0.1; proxy shows large logloss gains, 0.776 -> 0.70-0.77, but held-out proxy BA only -0.5..+0.9 and not monotone in lam) 3142573-74, scorer afterok.

## Result (Fri 20 Nov ~14:20 CET; Galvani waves 3142238-45, 3142518-21, 3142573-74, within0 nest rerun 3142789-91; table `laneR1/report.txt`, cells `laneR1/cells_all.json`)
All nested forms vs ds_all (both orders, people 80 %; people+orders in report.txt): nest_both s0.5 -0.087 [-0.22,+0.05], s1.0 -0.220; nest_self -0.43/-0.60; nest_other -0.45/-0.58 (full0 -0.44/-0.40, within0 -0.46/-0.76). Un-nested optimum (3 moves: '2' x2, '4' x2, '1' x0.5) +0.14 [-0.02,+0.32] (full0 +0.30, within0 -0.03): none of it survives selection. Cheap nested single-move estimate +0.04. Proxy-learned weights (lam 0.1/0.03/0.01/0.003): +0.05 / -0.38 / -1.05 / -2.21. Best single coordinate m_4 x2 +0.27 [-0.09,+0.65] (chosen on the scored data, not honest). Extras: prat/reve at 1 unit +0.20 each on full0 (inside noise), cbx and xfam negative.
Reading: the fixed mixer weights are not an absorber; stock weights are a local optimum, re-weighting gives nothing nested-honest. Nothing reaches +0.3 over ds_all, so no weights-only package. Undecided/not closed for weights; the ds_all gain stays with the components. Note: laneZ reaches the same on hardcb (lane Z score_z.md).
