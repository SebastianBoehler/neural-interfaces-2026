# Lane B7 · is lane P's joint arm seed-robust? (registered Fri 20 Nov 2026 ~13:10 CET, before any B7 arm was scored)

Question (BACKLOG B7): the best lane P arms (`ds_all` = hardcb + lite5 + FBL2 + dispersion correction, per-session reset, +0.86 pp vs cap250, +0.47 vs final stack; `ds_l5`, `hcF_l5`, also `ftlite5`) were each scored with ONE seed of the hard pair mean, ONE lite5 fit and ONE dispersion member (seed 0); the CBraMod slot is the 3-seed mean. Does the gain survive other seeds?
Code `laneB7/chain/` (final_b7.py = lane P chain, arm set and seed set selectable; score_b7.py; worker.sbatch), fit variants `laneB7/ttt_core.py` (lite5f1/lite5f2 differ from lite5 only in the torch/augmentation generator seed 1/2), segment export = `laneP/seg_export.py` seeds 1, 2 (same recipe, N9 member). Release `ni26_laneB7_20261120` (Galvani). Instrument as lane P: ten reused held-new people, orders full0 + within0 (the only pinned deliveries), official BA.

## Seed sets (one factor changed from lane P's base B0 = hard pair mean, CBraMod 3-seed mean, lite5 seed 0, member seed 0; plus three joint sets)
| set | hard | CBraMod | lite5 fit | member |
|---|---|---|---|---|
| B0 | pair mean | mean of 3 | s0 | s0 | (must equal lane P bitwise: zero-change guard)
| cb0 cb1 cb2 | pair mean | single seed 0/1/2 | s0 | s0 |
| h0 h1 | single hard seed 0/1 | mean | s0 | s0 |
| l5f1 l5f2 | pair mean | mean | fit seed 1/2 | s0 |
| seg1 seg2 | pair mean | mean | s0 | seed 1/2 |
| J0 J1 J2 | 0 / 1 / 0 | 0 / 1 / 2 | s0 / f1 / f2 | 0 / 1 / 2 | (all factors changed together)
Arms per set: stock, hardcb, ftlite5, hcF_l5, ds_l5, ds_all (+ hard, identity as guards). FBL2 doubling, constants, MASS 5/6, correction rule: lane P unchanged. Not varied (no alternative exists): the cap250 members, the FT base stream seed (lite5 only changes the fit generator), FBL2.
Caveat stated now: the single-hard-seed sets change the final stack itself (hard pair mean -> one member), so the pair-mean base is the pinned reference; every delta is reported vs cap250, vs the same set's hardcb and vs the pinned hardcb (B0).

## Statistics
Spread of each arm's delta across the 12 sets: min, max, SD (sets are not independent samples: one factor at a time). Pooled estimate = mean of the delta over sets (all 12, single-factor 9 + B0, joint 3); 80 % interval from 10,000 draws that resample the ten people AND the two orders (2 with replacement), paired over sets and arms. Dev-weighting caveat of AGENTS.md applies (unverified assumption of 57 % new people not used here: all numbers are held-new only, returning not scored).

## Predictions (pp, both-order mean)
ds_all vs cap250: B0 +0.86 (measured); pooled over sets +0.60 [+0.1,+1.1] (winner's-curse shrinkage of the three-way interaction); range across sets 0.6 (SD 0.2). vs same-set hardcb: pooled +0.25 [-0.2,+0.7]. ds_l5 vs cap250 pooled +0.45, range 0.5; hcF_l5 +0.45, range 0.5; ftlite5 +0.40, range 0.4 (CBraMod seeds spread ±0.15 in B1 S; lite5 fit seed ±0.2; member seed ±0.2). Zero-change: B0 equals lane P `full0_P.npz`/`within0_P.npz` terminal bitwise for all shared arms (job fails otherwise).
## Rule (stated before the run; reading aid, not a gate)
SEED-ROBUST for an arm: all 12 deltas vs cap250 > 0 AND the pooled people+orders 80 % lower bound vs cap250 > 0 AND pooled point vs same-set hardcb > 0. SEED-FRAGILE: any joint set (J0-J2) <= 0 vs cap250, or pooled point vs cap250 below +0.2 (half of hardcb's +0.39). Otherwise UNDECIDED. A vs-hardcb interval containing 0 is undecided, not closed (AGENTS.md).

## Result
(pending)
