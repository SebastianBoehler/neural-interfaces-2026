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
