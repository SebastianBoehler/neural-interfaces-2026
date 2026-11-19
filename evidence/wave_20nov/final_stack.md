# Final-upload candidate: cap250 + F-C25 hard + CBraMod slot member

Registered Thursday 19 Nov 2026 ~22:30 CET, before any stack arm was scored. Rules (docs/STATE.md, 19 Nov): only the best
sealed score counts, a Failed run costs no quota, budget 3,600 s on one RTX PRO 4500. Free option: pick the largest expected
gain that fits the runtime, not a do-no-harm gate. Never uploaded from this thread.

## Arms (all against exact cap250, full current system, shuffled delivery)
- **hard**: F-C25 hard pair replaces the five outer raw FBCNet members (MASS 5/6, existing c25d arm, re-run here as control).
- **cb**: CBraMod-C 3-seed ensemble in 1/6 of the raw slot (MASS 1/6, laneC member form), plus within0 (new delivery).
- **hardcb**: slot proposal `0.8*hard + 0.2*cb` at MASS 5/6 (= hard on four of the five replaced outer members' mass and
  CBraMod on the fifth; equals the package rule `mix + sum_i w_i (0.8*hard + 0.2*cb - member_i)`, w_i = 1/90).
- Instruments: (1) dev held-new P01-P10 via the corrected Friday chain `friday/cap250c25d` consumer, orders full0 and within0
  (the only two the chain supports; CBraMod LOPO posteriors for within0 are re-inferred from the saved LOPO fits through the
  package member code, which also gives the full0 bank-parity check); (2) returning P04/P10 via the Lane A worker where it
  accepts external raw-slot posteriors; (3) NI-V exact-package arms on Scherer (B-new; B-ret) with a Scherer-free (arm B)
  CBraMod all-public ensemble (the arm-C fits saw Scherer windows in stage A, so they would leak on NI-V).
- Weighting: sealed = 0.57 new + 0.43 returning, new = dev held-new, returning = P04/P10 arm (covers about 13 % of the
  sealed returning windows, stated with every number). Paired 80 % intervals resample people (and orders where several).

## Predictions (pp over cap250, balanced accuracy)
- Held-new full0: hard +0.43 (measured, control), cb +0.26 (measured), hardcb +0.55 [-0.05, +1.15]; interaction
  (hardcb - hard - cb) -0.10 [-0.5, +0.3] (F-C22 found a negative interaction between two raw-slot edits).
- Held-new within0: hard +0.37 (measured), cb +0.2 [-0.3, +0.7], hardcb +0.5 [-0.2, +1.2].
- Returning P04/P10 (0.2/0.8): hard 0 [-0.5, +0.5], cb 0 [-0.6, +0.6] (earlier CBraMod returning cells were weak: 52 % standalone).
- NI-V Scherer B-new: hard +0.1, hardcb +0.2 [-0.4, +0.8]; B-ret about equal.
- Package: parity of the packaged CBraMod posterior against the saved bank max |diff| <= 1e-3 (GPU fp32 kernels differ; bank
  guard used 5e-4); package terminal decisions equal the chain's hardcb terminal decisions on the public fixtures.
- Runtime: cap250 2,555 s sealed; hard x1.06 warm; CBraMod 3 forwards per window adds an estimated 10-25 % -> projected 3,000-3,350 s.

## Selection rule (fixed now)
Pick the arm with the highest sealed-weighted mean delta among arms that satisfy (a) held-new delta >= 0 in both orders,
(b) NI-V B-new delta >= -0.3 pp (a veto only; NI-V magnitude is not trusted: 22 of 41 channels interpolated),
(c) projected RTX time <= 3,300 s (8 % margin to 3,600 s) with parity and zero-change checks passed.
If hardcb fails (a) or (c) fall back to the best of hard / cb; if none qualifies, stay with cap250. Intervals crossing zero
are "undecided", reported, and do not by themselves veto (AGENTS.md; bars are reading aids).
