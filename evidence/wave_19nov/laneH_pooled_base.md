# Lane H · Pooled-corpus base for the cap250 raw-FBCNet family (same recipe, same seeds, with vs without external people)

Registered Thursday 19 Nov 2026 ~19:30 CET on Leonardo (account AIFAC_P02_1675), before any lane H posterior was scored.
The only things read before registering: recipe sources and the lane C / C2 cards (REVE B vs C: 55.0-55.4 vs 55.0-55.7 held-new).

## What "the cap250 neural cross-person family" is here (audited, not assumed)
cap250's frozen neural nets are SessionBN nets trained on all-public PROTEUS: `fb_sbn_rot05`, `fblight_ea_rot05_sbn`,
`fbcnet_raw_sbn` (net 48, `sbn_fbcnet_raw_k20_s1.pt`, the member the F-C20 raw swap replaces), plus
`sincshallow_ea_ptm7r_rot05_sbn` (already initialised from the 7-corpus m7r trunk) and the EEGNeX/IFNet teacher trunks (m7r,
frozen). So external pooling is already in the Sinc and teacher parts; the fb / fblight / fbraw SBN family is public-only.
Closed earlier and NOT repeated: more MI corpora for the teacher (m13r/m14r, WAVE10_3NOV), co-training (OVERNIGHT_2NOV).
Lane H tests the untested part: pooled trunk for the public-only SBN family member, with task-matched corpora (Scherer
SUB/WORD, Shin 2018 WG) in the pool. Scope of this card: ONE family (raw FBCNet, 5 of 6 frozen raw seeds in the slot); fb and
fblight are not covered and nothing here speaks for them.

## Arms (identical net, optimiser, 40 epochs, SessionBN k_train 20 / k_test 20, seeds 0-2; only the trunk differs)
- **A** `fbcnet_raw_sbn`: shipped recipe, public PROTEUS only (control).
- **B** `fbcnet_raw_pthp7_sbn`: trunk = shipped m7r pretraining (7 corpora, one head each, rot 0.05, 30 ep) -> `hp7`.
- **C** `fbcnet_raw_pthp9_sbn`: trunk on those 7 + Scherer 2015 + Shin 2018 WG/baseline in ONE task-matched 4-way block
  (motor / subtraction / word / rest), head transferred -> `hp9`. Scherer/Shin 2018 are external: no held-new leak.
Code `laneH/` (`pretrain_h.py` wraps experiments/pretrain.py unchanged; `lopo.py`; `collect.py`), launcher
`scripts/cloud/leo_lane_h.sh`. Folds: all_loso P01-P10 (the person's rows never trained on), plus an all-public deployment fit per
arm and seed. Delivery: the eight pinned full-session orders (`runs/shuffle_perms.npz`, sha 6046bf5b...), each held session
streamed window by window in delivery order (SessionBN causal prefix; batch == 3-window-prefix asserted).
Silent-no-op guards: arm B/C assert the trunk keys loaded; the pretrain wrapper asserts the new corpora were read only for hp9;
collector refuses a missing/partial fold set, wrong receipt, wrong order or wrong train rows.

## Predictions (balanced accuracy, held-new, 3-seed ensemble, person-and-order resampled 80 % in brackets)
Standalone: A about 68 % (no measured reference; the 3-seed REVE member is 55, cap250 73.7 full0). B-A **+1.5 pp [-1.0, +4.0]**;
C-B **+0.5 [-1.5, +2.5]**. Pooling gave REVE ~0 (lane C), but small trunks gained +2.9 dev when the head was the transferred part.
Full cap250 (laneH/full_replay.py, full0, F-C20 form: the five frozen raw seeds replaced by the arm's ensemble, MASS 5/6;
secondary: laneC form MASS 1/6): pooled arm vs exact cap250 **+0.15 pp [-0.5, +0.8]**, pooled arm vs the A control in the same
slot **+0.20 [-0.4, +0.8]**, A vs cap250 about 0 (control must be within +-0.4 of stock or the slot form is mis-specified).
Returning people: not scored (all-public fit exists only for deployment); stated as untested, not as zero.

## Rules (stated before scoring)
- Sealed-weighted claim = 0.57 x held-new paired delta (returning untested, counted 0). Bars are reading aids, not gates.
- **Supported** (build candidate): pooled-minus-A full-system paired 80 % lower bound > 0 on full0 AND B/C-minus-cap250 point > 0
  AND standalone B-A point > 0. **Refuted for this form/dose**: pooled-minus-A full-system 80 % upper bound < +0.3 pp AND
  standalone B-A upper bound < +1.0. Anything else **undecided** (instrument +-0.5 across people; neither P04 nor P10 decides).
- If supported by Fri ~13:00 CET: candidate ZIP = copy of cap250 package with the five fbcnet_raw seeds exported from the
  all-public pooled fits, `friday/qualify_package.py` runtime projection, NI-V B-new/A scoring before any upload claim.
- Not covered by any outcome: fb / fblight SBN families, k other than 20, more pooled corpora, MASS other than 5/6 and 1/6.

## Result (19 Nov ~20:00 CET; Leonardo 64870503 pretrain, 64870509 arm A, 64870510 arms B/C, 99 banks, all guards passed; Galvani 3137961-66 full0 consumer, dir `ni26_laneH_20261119`; files `laneH/`)
Smoke (64870002-04) passed first; each hp7/hp9 pretrain took ~8 min, each fit ~12 s. Held-new P01-P10, 3-seed ensemble, FULL-SYSTEM DEV cap250 stock 73.74 (cross-check equals c25d `stock|terminal`; zero arm bitwise equal).

| | standalone BA full0 / 8-order mean | B-A / C-A standalone, paired people+orders 80 % |
|---|---|---|
| A public-only | 59.31 / 59.29 | |
| B m7 pooled trunk | 59.81 / 59.80 | +0.51 [-0.39, +1.72] |
| C + Scherer + Shin 2018, unified block | 58.21 / 57.45 | -1.84 [-3.23, -0.44] (C-B -2.35 [-3.81, -1.11]) |

| Full cap250, full0, paired people-only 80 % | MASS 5/6 (raw family swap) | MASS 1/6 (laneC form) |
|---|---|---|
| A control vs stock | -0.151 [-0.37, +0.07] | 0.000 [-0.09, +0.06] |
| B vs stock | +0.038 [-0.15, +0.17] | -0.040 [-0.15, +0.09] |
| C vs stock | +0.108 [-0.09, +0.32] | -0.117 [-0.24, +0.05] |
| B vs A | +0.19 [-0.10, +0.45] | -0.04 [-0.14, +0.11] |
| C vs A | +0.26 [+0.06, +0.45] | -0.12 [-0.27, +0.10] |

Reading against the rules above: B is **undecided** (lower bound of B-A < 0, upper 0.45 > 0.3; standalone +0.5). C meets the lower-bound
clause only because the A control loses 0.15 itself; C vs stock is +0.11 [-0.09, +0.32] and standalone C-A is -1.84 with the interval below 0, so the
"standalone B-A point > 0" clause fails: **not supported**. Sealed-weighted claim for B: 0.57 x 0.04 = about +0.02 pp, nothing to build.
Both predictions were too optimistic standalone (B-A +1.5 predicted, +0.5 seen); full-system prediction for pooled vs stock (+0.15) was met in sign only for B/C at 5/6.
Not run: orders other than full0 in the full system, returning people, fb / fblight families, other k. No candidate ZIP built.
Why standalone gain stays small: the SessionBN fine-tune (40 epochs on public PROTEUS) overwrites most of the trunk; the unified 4-way head with Scherer's 22 interpolated channels hurts standalone.
