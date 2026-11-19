# Lane I · Selective shrinkage of SessionBN moments (rank 4 of research_codex.md)

Registered Thursday 19 Nov 2026 ~19:15 CET, before any shrinkage posterior was scored. Read first: `experiments/session_bn.py`,
the cap250 traces (`StreamBN` in `sbn_*_s*.pt`), `laneH/{lopo,collect,full_replay}.py`, §378 (`evidence/next_wave_16nov/shrink378.md`).

## What the current module is (audited, not assumed)
Every neural member of cap250 normalises with ONE SessionBN (`spatial_conv.1`, buffers rm/rv = source EMA moments, s1/s2/n = running
sums of the delivered windows' moments). Window t uses `mean = (k rm + sum_{i<=t} r1_i)/(k+t)`, same for the second moment: that already IS
age-only shrinkage toward the immutable source moments with weight n/(n+k), k a constant baked in the trace (k2, k5, k20 by family).
So S1 (age-only) = changing k (a re-tuned prior count), not a new mechanism. Prior art in the ledger: §378 replaced k20 by a source-fitted
per-channel k=W/T on the A6 body: standalone +0.42, full B2_pool -0.017 [-0.15,+0.14]; this lane tests the same operator on the raw FBCNet
family in the cap250 chain, with a data-adaptive (S2) weight.

## Form (one high-mass family, stated limits)
Family = `fbcnet_raw_sbn` (`sbn_fbcnet_raw_k20`, 6 of cap250's raw slot; lane H arm A instrument): same LOPO fits, seeds 0-2, 40 epochs, then the
eight pinned full-session orders streamed window by window. Fit once per (seed, held person); arms differ ONLY in the eval-time moment rule.
Scored in the full system by `laneH/full_replay.py` (full0, 3-seed mean replaces 5/6 of the raw slot, MASS 5/6; secondary MASS 1/6).
NOT covered: fb (k2/k5), fblight, sinc families; the cap250 zero arm is the lane-H control A, which is not bitwise cap250 (retrained nets).
- **S0** k=20 (current behaviour). Zero checks: (a) the S0 bank regenerated here must match lane H's `ensemble_A_held_new.npz` (report max abs
  diff; same code, same seeds; GPU class may differ so equality is reported, not assumed); (b) full-system zero arm equals cap250 stock bitwise
  (replay guard, as lane H); (c) the S2 weight formula at a=n/(n+20) equals S0 within 1e-5 (unit test); S2 mode off delegates to S0 code.
- **S1** fixed age-only weight n/(n+tau), tau in {5,10,20,40,80,160}. tau for held person P is picked by nested LOPO: mean standalone
  BA (3-seed ensemble, 8 orders) over the nine OTHER people's own LOPO folds; ties and gains <0.1 pp go to 20. P's outputs never enter its tau
  (the other folds' nets did train on P's rows; stated limit).
- **S2** per-channel risk-weighted weight a_c,t = n T_c / (n T_c + W_c,t). T_c = between-session variance of session-mean window moments,
  W_src_c = mean within-session variance of window moments, both from the training people's windows only (hook on the trained net's SessionBN
  input, fold person excluded). W_c,t = (4 W_src_c + sum_{i<=t}(r1_i - m_t)^2)/(4 + n) from delivered windows only (class-blind, past
  and current window as in the stock formula). Same a_c,t for first and second moment. No labels anywhere.
Added cost: S1 none (a constant). S2 a few element-wise ops on 32-feature vectors in the BN: measured by microbenchmark, target < 1 ms/call.

## Prediction (balanced accuracy, held-new, FULL-SYSTEM DEV full0, paired people 80 %)
Prior from `forecast_calibration.md`: +0.15 [-0.55,+0.85] for the card. Adjusted down by §378 (null at the full system, +0.42 standalone):
S1 minus S0 **+0.05 [-0.5,+0.6]**, S2 minus S0 **+0.10 [-0.5,+0.7]**; standalone S2-S0 +0.3 [-0.5,+1.1]. Adaptive weights have
been over-predicted in magnitude (AD lever, calibration table), so no step change expected. Sealed-weighted claim = 0.57 x held-new (returning
counted 0 unless scored; the returning arm needs a patched package through Lane A's scorer, started only after the held-new result).
## Rule (stated before scoring)
**Refuted for this form**: upper 80 % (people-and-order resampled for standalone; people-only for the full0 replay, one order) of arm minus S0 <= 0
on held-new full-system full0. Supported-for-build: lower 80 % > 0 on full0 and standalone point > 0. Else undecided (instrument +-0.5 across
people). Both arms are reported against S0 and against cap250 stock; P04 and P10 are reported but decide nothing.

## Result (19 Nov 2026, ~23:30 CET; Galvani 3138093 LOPO array, 3138147 collector, 3138148-53 full0 replays, 3138293 returning array, 3138294 collector; files `laneI/`)
Fits: 3 seeds x 10 LOPO folds, 7 eval rules each, 8 orders (23 s per fold). Guards passed: S2 formula at a=n/(n+20) equals stock k=20 to 5e-13 (float64);
causal prefix check on every session; zero arm equals cap250 stock bitwise in all six replays (replay guard); stock terminal argmax equals c25d `full0.npz`.
Limit on the S0 zero check: the regenerated S0 bank is NOT bitwise lane H's arm A (A100 vs 2080 Ti class; max abs 0.081, 39 of 2655 argmax decisions differ), so S0 is
the control in the slot (S0 vs stock -0.041 at 5/6, 0.000 at 1/6), not cap250 itself. Only the raw FBCNet family (k20) was changed; fb/fblight/sinc untouched.
**Standalone, held-new, 3-seed ensemble, 8 orders, paired people+orders 80 %:** S0 59.44; S1 minus S0 +0.13 [-0.73,+1.16]; S2 minus S0 -0.11 [-0.94,+0.84].
Fixed-tau grid minus S0 (not selected): k5 +0.25 [-0.68,+1.39], k10 +0.34 [-0.18,+0.89], k40 -0.56, k80 -1.29 [-2.46,-0.23], k160 -1.88. The nested rule picked tau=5 for nine
people and 10 for one: the optimum sits on the grid edge, so a shorter prior than the grid allows may be better standalone (not run; beyond the registered grid). S2 implies an
effective prior count W/T of about 63 (median T/W 0.016), i.e. longer than 20, which is the side that loses standalone.
**Full system, held-new P01-P10, full0 only (laneH replay; paired people-only 80 %, one order, so no order variance):**

| contrast | MASS 5/6 (raw family swap) | MASS 1/6 |
|---|---|---|
| S0 minus stock | -0.041 [-0.223,+0.130] | 0.000 [-0.087,+0.058] |
| S1 minus S0 | **-0.052 [-0.180,+0.071]** (P04 -0.20, P10 +0.13) | -0.113 [-0.228,+0.002] |
| S2 minus S0 | **+0.034 [-0.119,+0.160]** (P04 -0.02, P10 +0.35) | -0.112 [-0.223,0.000] |
| S1 / S2 minus stock | -0.093 [-0.318,+0.129] / -0.007 [-0.306,+0.257] | -0.113 / -0.112 |

Rule: refuted needs upper 80 % <= 0. S1 and S2 at 5/6 have upper +0.07 and +0.16, so **undecided, not refuted, and not supported** (lower bounds < 0; points inside +-0.5).
The 1/6 form has upper ~0 for both (-0.11 each), leaning refuted in that slot. Prediction check: S1 +0.05 predicted, -0.05 seen; S2 +0.10 predicted, +0.03 seen (both inside the stated 80 %).
**Returning P04/P10 (Lane A instrument, 8 orders x visits 2-3, live package incl. the fine-tuned raw member 47, six raw traces rewritten to baked k):** only S1 exists here
(S2 needs per-trace source moments, not built). k20 control equals stock bitwise on all 32 cells; the pinned D-EArW blob proof for P10 was replaced by an assertion that the
fine-tuned blob is the patched file. k10 minus stock -0.053 (20/80) [order 80 % -0.093,-0.020], k5 -0.091 [-0.137,-0.046] (P04 -0.198, P10 -0.065), 3 of 8 orders non-negative.
Two people only: descriptive. Recognition ages and gates were identical in the logged cells.
**Sealed-weighted S1 (k5 for returning, tau chosen for new): 0.57 x (-0.052) + 0.43 x (-0.091) = -0.07 pp; S2: 0.57 x (+0.034) = +0.02 (returning untested).** Intervals cross zero where
they can be formed; the new and returning numbers come from different instruments, so no joint people+orders interval is claimed.
Cost: S1 zero (a baked constant). S2 microbenchmark (eager torch, 1 CPU thread, one 32-channel BN call): 0.029 -> 0.038 ms, so about 0.05 ms for six raw members;
not measured on the RTX worker. Not run: a second full order (the replay is hard-wired to full0), the fb/fblight/sinc families, the permuted-source control, tau below 5.
Reading: the stock prior count is near the standalone and full-system optimum; the shrinkage form adds no consistent positive. Nothing to build from this lane. Same conclusion as §378.
