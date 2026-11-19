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
