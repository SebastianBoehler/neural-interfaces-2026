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

## Results (19 Nov, jobs below; all arms in the full chain, paired 80 % intervals; pp over exact cap250)

Held-new dev, ten reused people (`submissions/final_stack/chain`, release `ni26_final_stack_20261120`; chain jobs 3139080 full0, 3139081
within0, scorer 3139383; CBraMod bank streamed through the package member code, 3138834/3138835; guards: identity arm equals stock
bitwise, stock and hard arms equal the F32 parent banks bitwise, member moves mix/rebalance/EM/TSEM):

| arm | full0 | within0 | both orders |
| --- | ---: | ---: | ---: |
| hard | +0.331 [+0.214, +0.448] | +0.305 [-0.005, +0.642] | +0.318 [+0.178, +0.469] |
| cb (1/6 slot) | +0.263 [+0.004, +0.529] | -0.004 [-0.168, +0.207] | +0.130 [0.000, +0.302] |
| hardcb | +0.514 [+0.336, +0.697] | +0.270 [+0.112, +0.434] | +0.392 [+0.280, +0.519] |
| interaction hardcb-hard-cb | -0.080 [-0.228, +0.076] | -0.031 [-0.391, +0.275] | -0.055 [-0.205, +0.074] |

CBraMod bank parity (package member vs Lane C2 bank, full0, 10 held people, seeds 0-2): max |diff| 3.1e-6, 5.7e-6, 3.8e-5
(fp32 weights); fp16-stored weights shipped in the package: 2.3e-4 through the official Solver (P02). Prediction held (hardcb +0.55,
interaction -0.10, cb within0 +0.2 was optimistic: 0.00).

Returning P04/P10 (Lane A instrument, 8 orders, source-fold packages built with the same StackEnsemble code; hard actors are the
returning F17/C20 actors that exclude the whole person, CBraMod = returning fits excluding P04/P10 visits 2-3; job 3138858): w20/80
hard -0.020 [-0.083,+0.044], cb +0.050 [-0.020,+0.118], hardcb +0.011 [-0.072,+0.096]; equal cells -0.100 / +0.032 / -0.067; P04
hard -0.234. This covers about 13 % of sealed returning windows and its hard actors are not the F32e pair: weak evidence.

NI-V Scherer (RTX 2080 Ti exact packages, hard = F-C25p ZIP, hardcbB = hard + Scherer-free CBraMod B-arm all-public fits trained on
Leonardo 64883819/20, since arm-C fits saw Scherer): B-new full-session hard -0.065 [-0.26,+0.13], hardcbB -0.058 [-0.28,+0.15];
B-ret hard -0.023 [-0.20,+0.16], hardcbB +0.107 [-0.10,+0.32]; NI-V sealed-weighted hard -0.047 [-0.18,+0.09], hardcbB +0.015
[-0.14,+0.17]. Null (cap250 seed 1) is about -0.08, so nothing here is outside noise; no veto (threshold -0.3).

Sealed-weighted (0.57 held-new both orders + 0.43 returning 20/80 P04/P10): hardcb +0.23, hard +0.17, cb +0.10.

Runtime (friday/qualify_package.py, warm-up replay, ref/cand x2): RTX 2080 Ti cap250 165.8 s, stack 208.4 s (ratio 1.257), A100 146.5 vs
180.0 (1.229); projected sealed 2,554.9 s x ratio = 3,211 s (RTX), 3,140 s (A100): 389 s margin to 3,600 s (10.8 %); the ratio comes
from a short replay, the sealed per-window share may differ and the RTX PRO 4500 is untested. Public replay score identical to
cap250 (0.994578853046595) on both devices, files unchanged, causal/reset/prefix/future checks pass, no [v6] errors.
Package 90.5 MB (self-imposed 90 MB allowance exceeded; qualification ran with NI26_MAX_MB=100).

Selection rule applied: hardcb passes (a) held-new >= 0 in both orders, (b) NI-V >= -0.3, (c) projected 3,211 s <= 3,300 s.

## Review fix · dead-session crash (20 Nov ~23:00 CET)
Bug: after any v6-chain error, step62 returns v5.1b's output without mix6, so `HardWaveEnsemble.step` raised "the actual
outer mixer did not consume the current waveform" outside step62's try and ended the whole run (stock cap250 survives
the same injected error). Repro: one injected `eye.step` error at window 4 of P01/1 -> CRASH (hardcb), survives (stock).
Fix: guard skipped when `self.dead`; `submissions/final_stack/patch_dead.py` derives
`runs/sealed/final_zips/final_stack_hardcb_safe.zip` sha 8a868f22…80f7 (only submission.py differs). Checks: injected
error now falls back and the session continues; no-error labels identical to hardcb on P01/1 (190) and P04/2 (189)
windows, CPU. Not yet rerun: RTX/A100 qualification (code path unchanged when no error occurs). Upload hardcb_safe, not hardcb.
Same guard lives in every hard* trial ZIP (tttlite5, tttsparse20, …): patch those with the same script before use.

## Qualification of final_stack_hardcb_safe.zip (21 Nov, friday/qualify_package.py, NI26_MAX_MB=100, ref cap250)
sha 8a868f22…80f7 (94.9 MB, 44 members). Local (Mac CPU): sha/CRC/layout OK, files unchanged, public replay 0.994578853046595
(= cap250 = hardcb), reset/prefix/future-perturbation/labels checks pass. Cluster (Galvani worker SIF, ref/cand x2; first two
jobs 3144783/3144784 died in 1-5 s on a dangling ref.zip symlink, resubmitted with a real copy): A100 job 3144789, RTX 2080 Ti
job 3144790. Public score identical to cap250 and hardcb in all 8 replays, files unchanged, all 24+8+8 [v6] lines `errors 0`.
Runtime (min of 2): A100 ref 129.1 s, cand 160.8 s, ratio 1.2455 (hardcb 1.229, +1.3 %), projected 3,182 s; RTX 2080 Ti ref
137.2 s, cand 168.1 s, ratio 1.2252 (hardcb 1.257, -2.5 %: faster, absolute times differ from the hardcb run because the nodes
differ), projected 3,130 s. Margin to 3,600 s: 418-470 s. The script's QUALIFY line reads FAIL only because of two self-imposed
checks: projection <= 3,060 s (0.85 x 3,600; hardcb's 3,211 s misses it equally) and a repeat-score comparison that includes
the `duration` field (scores themselves are identical). RTX PRO 4500 still untested.
