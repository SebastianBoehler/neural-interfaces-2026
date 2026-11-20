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
