# Lane T · team C's TTT recipe inside the final stack's fine-tuning stage (registered Fri 20 Nov 2026 ~08:15 CET, before any arm was scored)

Question: the final stack (cap250 + F-C25 hard + CBraMod slot; `final_stack.md`) fine-tunes four SessionBN families (raw FBC,
EA FBC k2, Sinc, FBLight; seed 0 each) in the session: first refit at window 20, every 10, five AdamW steps (lr 1e-3) on the
whole prefix, hard relabelled targets weighted by confidence, KL(clean || strong view), refits stop at the first age >= 250.
Does team C's recipe (Sinkhorn-balanced soft targets, information maximisation, augmented views, more steps) do better there?

## Instrument (what is and is not new)
- Held-new dev, ten people, orders full0 and within0 (the only two the corrected Friday chain supports; two FULL-shuffle orders
  are not available in it, stated). The chain freezes the inner route, EM/TSEM gates and recognition and re-runs mix, rebalance,
  EM, TSEM, count and CSP; the FT family posteriors `CU17[family,0]` are the only input changed. FT targets are the chain's own
  per-refit relabelled posteriors from the stock run (not re-derived from the changed posteriors): the label loop is frozen,
  as for every other chain arm. No cap250 LOPO Solver packages exist, so a real-package held-new run is not possible.
- New FT banks come from the restored LOPO fold checkpoints with the same pinned actor code (`ttt_core.py` replaces only
  `run_stream`; `laneT/fit.sbatch`, copies of the pinned matched-fit release per arm). Arms: `ctl` = the unmodified stock stream
  (zero-change: must reproduce the released member banks), `light`, `heavy`.
  - light: stock schedule, optimiser, steps and strong view; hard CE -> soft CE to Sinkhorn(10) balanced target + 0.3 * IM.
  - heavy: 20 steps, per-window strong views (shift 15, scale 0.15, channel drop 0.1, relative noise 0.2), loss on <= 64 random
    prefix windows per step (full-prefix forward because SessionBN is causal in window order), IM 0.3, L2 anchor 1e-2 to source.
- Chain arms (all MASS 5/6, proposal 0.8 hard + 0.2 cb, i.e. the final stack): `stock_ft` (= final stack, control), `light`, `heavy`.
- Returning P04/P10 and runtime: real package copies (new dir, `submissions/laneT/`), FT class swapped; P04/P10 recognise at
  age 10-60 and recognition retires FT permanently, so little effect expected there.

## Predictions (pp balanced accuracy over the final stack, held-new, mean of the two orders)
light +0.10 [-0.35, +0.55]; heavy +0.20 [-0.45, +0.85]. Returning P04/P10 (0.2/0.8): light 0 [-0.3, +0.3], heavy 0 [-0.3, +0.3]
(FT is retired early). Zero-change: `ctl` banks equal the released member banks (max |diff| <= 2e-5, argmax changes 0); the chain
stock_ft arm equals the final22 hardcb arm bitwise. Runtime: light about equal to stock FT; heavy about 4x FT compute, projected
sealed time 3,211 s -> 3,3xx s (prediction: stays under 3,600 s).

## Rules (fixed now)
Per order, people-resampled 80 % paired interval (10,000 draws) of arm minus stock_ft: supported if the lower bound > 0 in both
orders, refuted if the upper bound < 0 in both, else undecided (bars are reading aids inside +-0.5 noise; neither closes the
direction). Adoption for the fallback package needs: both-order mean delta >= +0.2, returning not below -0.1 (20/80), projected
runtime <= 3,300 s from the real-package timing, zero-change checks passed. If heavy does not fit, the strongest form that
fits (fewer families, fewer steps) is registered as a new arm before it is scored.

## Result (Fri 20 Nov 2026 ~09:30 CET; files `evidence/wave_20nov/laneT/`; code `laneT/`, `submissions`-free: packages built by `laneT/build_ttt.py`)
Jobs (Galvani, release `ni26_laneT_20261120`): member streams 3140361-3140384 (2 orders x ctl/light/heavy x 4 families; each family 1-15 min),
chain smoke 3140447, chain complete 3140448 (full0) and 3140449 (within0), score 3140458, returning 3140478 (32 tasks) + collect 3140557,
public-replay timing 3140454-57, unrecognised-session timing 3140514/3140515. Guards: all 24 member jobs passed the pinned actor checks
(frozen base parity, prefix causality on the smoke cut, active refit); ctl (unmodified stream) reproduces the released member banks bitwise
for fbraw/fb/sinc and within 2.5e-4 for fbl (GPU kernels), and the chain arm `ftctl` equals the final stack's hardcb terminal labels
(contrast exactly 0.000 in both orders; raw scores differ <= 1.5e-5). Instrument limits as registered: full0 and within0 only (no second
full-shuffle order is available in the corrected chain), frozen inner route and FT targets, ten reused people.

Held-new dev, pp over the final stack (hardcb), paired people-resampled 80 % intervals (10,000 draws):
| arm | full0 | within0 | both orders |
|---|---:|---:|---:|
| TTT-light | +0.492 [+0.118, +0.796] | -0.335 [-0.669, +0.009] | +0.079 [-0.169, +0.314] |
| TTT-heavy | +0.651 [-0.009, +1.230] | +0.034 [-0.371, +0.452] | +0.343 [-0.127, +0.783] |
Absolute (both orders): cap250 76.03, final stack 76.42, light 76.50, heavy 76.77. Per-person deltas are noisy (heavy full0 from -1.10 to +2.68;
within0 -1.06 to +1.60), so by the registered rule both are undecided (lower bound > 0 in one order only for light, none for heavy).
Predictions: light +0.10 (seen +0.08, within0 negative), heavy +0.20 (seen +0.34, upper side of the interval).
Returning P04/P10 (real fold packages, 8 orders x 4 cells, `ret/`): FT ran in only 10 of 32 cells (recognition at age 10-60 retires it,
4 refits at most). w20/80: light -0.040 [-0.079, 0.000], heavy -0.101 [-0.208, 0.000] (order-only interval); equal cells -0.050 / -0.051;
P04 -0.066 / +0.032, P10 -0.033 / -0.134. Predicted 0 +- 0.3: held (small, not distinguishable from zero with two people).

## Runtime (measured partly; the Galvani connection dropped before the heavy numbers could be read)
Public-replay ratio vs cap250 (qualify_package.py, ref/cand x2, first uploads of the lane T zips): light 1.227 A100 / 1.299 RTX (projected 3,136 / 3,318 s),
heavy 1.238 / 1.259 (3,164 / 3,217 s); the final stack was 1.229 / 1.257. This replay is NOT informative about FT cost: the public people are recognised
at window 10-60, which retires FT, so almost no refits run (every cand scores the same 0.994578853046595).
Unrecognised sessions (`laneT/time_ft.py`, white-noise windows so recognition never fires, 190 windows, official fresh_model path, RTX 2080 Ti / A100):
final stack 278.2 / 260.0 ms per window (FT 12.2 / 7.6 s per session, 17 refits, 340 steps), TTT-light 276.9 / 259.2 (identical cost). TTT-heavy (4x the
steps per refit, same refit count and same prefix forwards) should add about 3x the FT seconds, i.e. +23 to +37 s per unrecognised session; at about 60 new-person
sessions in the sealed set that is +1,400 to +2,200 s on top of 3,211 s, i.e. about 4,600-5,400 s projected: heavy does NOT fit 3,600 s (estimate; jobs 3140514
(RTX) and 3140515 (A100) hold the measured heavy numbers in `ni26_laneT_20261120/time/full-3140514.log` and `full-3140515.log`).
TTT-light costs the same as stock FT, so its sealed projection is the final stack's (3,211 s, margin 389 s); it is also the only form that fits as it stands.

## Fitting forms, registered Fri ~09:40 CET before any is scored (not run: cluster connection needs the user's 2FA)
Budget: at most +89 s over the stack's 3,211 s (3,300 s bound), i.e. at most the stock FT cost per unrecognised session. Arms, both with the stock refit count
or fewer, same targets (Sinkhorn-balanced relabelled posteriors, IM 0.3, per-window views, <= 64-window loss subset, anchor 1e-2):
 - `lite5`: heavy objective at stock cost, 5 steps every 10 from 20 (isolates objective and views from step count);
 - `sparse20`: 20 steps every 40 from 20 (refits at 20, 60, ..., 220: 6 refits x 20 steps = 120 steps vs stock 115).
Predictions (both orders, vs final stack): lite5 +0.15 [-0.4, +0.7]; sparse20 +0.20 [-0.45, +0.85]. Same reading rule. `ttt_core.ARMS` needs `every` made per-arm
(sparse20 changes the relab grid lookup: target grid index (t - 20) // 10 stays exact because 40 is a multiple of 10).
