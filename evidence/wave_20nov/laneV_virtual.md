# Lane V - virtual subjects for the raw FBCNet SessionBN family (cross-person coverage by synthesis)

Registered Thursday 19 Nov 2026 ~21:55 CET (Ferranti H100 queue was ~2 days deep at submission; run on Leonardo AIFAC_P02_1675, array 64884082), before any lane V posterior was scored.
Read first: AGENTS.md, audit_20nov/SYNTHESIS.md, first_principles_fable.md (R1/T-B), laneH_pooled_base.md, forecast_calibration.md.
Diagnosis tested: new people carry ~75 % of sealed error and the source-people curve is steep (7 to 9 people +2.3-3.8 pp standalone), so
synthetic extra "people" might buy part of that slope. Prior art in the ledger: N12 mixup -0.7 standalone (core1, other net, recording
order) and `cmix`/`fsc` never applied in SessionNet (session_bn.py assert); arm B's config (fbcnet_raw_pthp7_sbn) has NO per-window rotation (rot 0, asserted), so V2 adds the rotation family at chunk level.
So V1 is a new form (cross-PERSON same-class mixup inside SessionBN chunks), not a repeat.

## Arms (identical to Lane H arm B: hp7 trunk, SessionBN raw FBCNet, 40 ep, k 20, seeds 0-2, LOPO P01-P10; only the training windows differ)
Code `laneV/` (vaug.py, lopo_v.py, collect_v.py, fit.sbatch); `laneH/lopo.py` and the hp7 trunks (sha-checked copies from Leonardo) unchanged.
Windows reach the net through a tensor stand-in on SessionNet._train; fit() asserts the augmentation ran and changed windows, V0 asserts it is absent.
- **V0** control = arm B, re-run on H100 (A100 Leonardo bank exists; hardware-matched control is V0).
- **V1** each window w.p. 0.5 -> lam x + (1-lam) x', x' random same-class window of a different training person, lam ~ U[0.5,1], rescaled to the window's power.
- **V2** one virtual person per 32-window SessionBN chunk: M = D Q S (electrode shift S: I + K(delta) - K(0), inverse-distance kernel on an approximate 10-10 grid,
  delta ~ N(0, 0.35^2) grid units; rotation Q = expm(A - A^T), A ~ N(0, 0.15^2); channel gain D = exp(0.10 N)); chunk rescaled to its power.
- **V2s** same, strong dose (shift 0.7, rot 0.30, gain 0.20) - second form, registered now, not chosen after seeing results.
- **V3** V1 then V2 (default dose). Doses are fixed here; no tuning.
Scoring: standalone held-new LOPO, 3-seed ensembles, eight pinned orders (collect_v.py: arm minus V0, people-and-order resampled 80 %).
Full system: `laneH/full_replay.py` on Galvani (full0), MASS 5/6 (raw-family swap) and 1/6, each V arm vs exact cap250 and vs V0 (paired, people-only 80 %).
Reference numbers: A 59.3, B 59.8 standalone; full system A-stock -0.15, B-stock +0.04 (5/6), -0.04 (1/6). Returning people: not scored (stated untested).

## Predictions (pp; standalone arm minus V0, 80 % in brackets)
Standalone: V1 +0.2 [-1.0,+1.4]; V2 +0.5 [-1.2,+2.2]; V2s +0.4 [-2.0,+2.5]; V3 +0.5 [-1.5,+2.5]. (Real people give +3/2 persons; synthetic ones assumed worth ~15 %.)
Full cap250, arm minus V0 (forecast_calibration rule: standalone x 0.1, widened to the instrument +-0.3): V1 +0.02 [-0.30,+0.30], V2 +0.05 [-0.30,+0.40],
V2s +0.04 [-0.35,+0.40], V3 +0.05 [-0.35,+0.45]. Arm minus cap250 = V0-minus-cap250 (about 0 +- 0.2) plus the same increments.

## Rules (stated before scoring)
- **Refuted for this form/dose**: full-system (arm minus cap250, 80 % people interval) upper bound <= 0 at 5/6 AND at 1/6.
- **Supported** (build candidate): arm-minus-cap250 lower bound > 0 at 5/6 AND arm-minus-V0 point > 0 AND standalone arm-minus-V0 point > 0.
- Otherwise **undecided** (instrument +-0.5 across people; neither P04 nor P10 decides; sealed claim = 0.57 x held-new delta).
- A candidate would need a package export from all-public fits (not done here) and joins the stack only via the final-stack rules.
- Not covered: other doses, other families (fb, fblight, Sinc), returning people, orders other than full0 in the full system.

## Result part 1 (20 Nov ~00:15 CET; Leonardo fit array 64884082, collector 64884654; all guards passed, V0 reproduces Lane H arm B: 59.80)
Standalone held-new, mean of 8 orders, arm minus V0 (people+orders 80 %): V1 -0.09 [-0.97, +0.68]; V2 -14.2 [-16.5, -12.1]; V2s -16.5 [-19.6, -13.9];
V3 -14.7 [-16.8, -12.6]. Predictions (+0.2 / +0.5 / +0.4 / +0.5) missed for V2/V2s/V3 by ~15 pp: the registered dose was far too strong
(A ~ N(0, 0.15^2) per entry on 41 channels is a large rotation; the shipped 0.05 is per window and a different scale; random 41-dim rotation scrambles the
lateralised spatial pattern the net needs). V1 is flat (undecided standalone).
## Added form after seeing V2 (registered before scoring it; not a rescue of V2): doses V2w / V3w = rot 0.05, gain 0.05, shift 0.15 (V3w = V1 + V2w).
Prediction standalone vs V0: V2w -0.3 [-2.0,+1.3], V3w -0.2 [-2.0,+1.5] (a rotation that small mostly equals the shipped rot05; expected near zero). Same rules as above.
Full system will be run for every arm that is not catastrophic standalone (V1, V2w, V3w); V2/V2s/V3 are closed at their registered dose by the standalone -14 to -16 pp
(no full-system run: a member that loses 14 pp cannot raise a 73.7 % system; stated, not measured).

## Result part 2 (20 Nov ~00:50 CET; Leonardo 64884827 fits V2w/V3w; Galvani 3138902-05, 3138973-76 full0 consumer, dir `ni26_laneV_20261120`; `laneV/paired_v.py`)
Standalone mean of 8 orders, minus V0: V2w -2.42 [-3.27, -1.70]; V3w -3.03 [-4.33, -1.98] (V1 -0.09 [-0.97, +0.68]). Even the weak chunk-level rotation/shift hurts.
Full cap250 full0, paired people-only 80 % (V0 itself reproduces Lane H B exactly: +0.038 at 5/6, -0.040 at 1/6; stock 73.74):

| arm | vs cap250, MASS 5/6 | vs cap250, MASS 1/6 | minus V0, 5/6 | minus V0, 1/6 |
|---|---|---|---|---|
| V1 | -0.234 [-0.566, +0.102] | -0.114 [-0.191, -0.043] | -0.272 [-0.629, +0.104] | -0.074 [-0.220, +0.060] |
| V2w | -0.048 [-0.438, +0.361] | -0.005 [-0.122, +0.156] | -0.086 [-0.438, +0.325] | +0.035 [-0.090, +0.171] |
| V3w | +0.139 [-0.179, +0.422] | +0.033 [-0.086, +0.182] | +0.101 [-0.185, +0.420] | +0.073 [-0.078, +0.226] |

Reading against the rules: nothing is **supported** (no lower bound > 0, standalone V2w/V3w minus V0 < 0). **Refuted by the rule** only where both masses have upper <= 0: none
strictly (V1 5/6 upper +0.10); V1 is the closest (negative at both). V2/V2s/V3 (registered doses) and V2w/V3w are standalone losses of 2-17 pp; the small full-system
V3w +0.10 is inside the instrument noise and unsupported by the standalone sign, so **undecided, no build**. Mixup (V1) is neutral standalone and slightly negative in the system.
Forecast check: standalone predictions missed by 2-15 pp (too optimistic, sign wrong for all spatial transforms); full-system predictions (+0.02 to +0.05) were inside their intervals for V1/V2w/V3w.
Not covered: dose between V2w and V0 (rot 0.02-0.03), time/frequency-domain virtual subjects, fb/fblight/Sinc families, returning people, all-public deployment fits.
Reading: random spatial transforms of the EA-whitened space destroy the lateralised pattern; cross-person same-class mixup adds no coverage. Synthetic people did not substitute for the +2.3-3.8 pp real-people slope.
