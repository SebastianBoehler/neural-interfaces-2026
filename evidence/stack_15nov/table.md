# s281 step 0: recorded small positives not in v6.6 (cap250, ZIP 9c4d5538, bundle 37c666a8)

Read-only table, 15 Nov 2026, built from docs/LEDGER.md results (no chain job run). v6.6 already ships: v5.1b stack, FT (gC5/r15 labels), eye view + rest member,
count filter (hard G1 groups, key v6_cf_on), T (personal TSEM prior) for recognised sessions, capped causal personalisation (v6_cp_on, cap 250), empty_cache refits.
"Gain" = chain-scored, new people, against the matching control, [80 % interval], people up. "Chain-scored" means through shufcombo; base-level numbers are listed separately.
Excluded (chain check negative): s272 eegn/conf (A -0.13/-0.21, B -0.69), s256 C10 fusion (-0.84), s265 V2 (-1.23; S2 form +0.03), s252/s254/s258/s259 label-noise routes, s241 weights (-0.26), s240 eye events (-0.05), s263 (-1.4 to -3.9), s270, s280 CBraMod (-1.30 to -0.04).

## A. Chain-scored mean > 0 (stack candidates)

| item | section | gain | interval / people up | schemes (within / full / runs) | run-time cost | code to switch on in the shufcombo chain | ready? |
|---|---|---|---|---|---|---|---|
| GSCF soft fingerprint neighbourhoods in the count filter (gamma/top-k nested) | s251 | full +0.30 (vs gated latch J) | [+0.06, +0.53] 6/10 | within +0.03, full +0.30, runs -0.02 | CPU only, a few ms per window (soft counts); no net forwards | studies/14nov/softfp/softcount.py (+ stack.py, score_off.py; shufcombo through runpy). Not in lean/src66 (package has hard G1 count filter only) | Chain-ready: yes (study code). Package: no. Only 1 of 3 schemes up and kernel curve sharp (gamma 0.5/1/2: +0.09/+0.30/-0.21); smooth form +0.16 [+0.04,+0.28] 8/10 full only. Undecided by the rule, fails the 2-of-3 scheme line |
| Eye-view copies EYEC 1 or 2 (extra copies of the eye column in SessionEM/TSEM) | s235 / s236 | +0.09 nested held-out | [+0.07, +0.11] over splits; 6-7/10 | within -0.02, full +0.16, runs +0.12 | none (a weight) | shufcombo env knob EYEC (studies/14nov/sweeps/, evidence/sweeps_14nov/nested_s235.txt); no key in lean/src66 (grep: none) | Chain-ready: yes. Gain inside the +-0.3 noise; scored on v6f, not on v6.6 |
| Template feedback ext/J (TSEM templates from relabelled windows) over the gated filter | s239 part 1 | within +0.19, runs +0.10 | [-0.05, +0.42] / [-0.09, +0.30] | within +0.19, runs +0.10, full not read | one extra template pass per refit, CPU only | studies/14nov/designdec/{stage,wrap}.py (runpy wrap of shufcombo; no-op check exact) | Chain-ready: yes. Kill rule met in s239 (< +0.3); intervals include 0; not in package |
| FT on run-urn smoothed labels (LPD) | s239 part 2 | no filter: within +0.07, runs +0.09; with the filter about 0 | within [-0.03,+0.17]; runs [-0.01,+0.21] | full not scored | same FT cost as r15 (labels differ only) | studies/14nov/designdec/{score2,ftrel_dd}.py (FT_V); killed in s239 | Listed for completeness; about 0 with the shipped filter, so not a stack candidate |
| Drop fb or fbraw (free drops, s273) | s273 | raw +0.09 each, filtered +0.02 / +0.01 | [-0.15,+0.30] / [-0.14,+0.33]; 7/10 / 6/10 | per scheme in evidence/ablation_15nov/s273.txt | NEGATIVE cost (a time budget): fbraw 1 : sinc 0.64 : eegn 0.5 forward-cost ratios; fb and fbraw each about 22 % of forwards | studies/14nov/newmember/run.py ARM=D DROP=fb|fbraw (run_c.py takes a comma list, s276) | Chain-ready: yes. A time budget, not a gain; s276 compound arms (C1-C4) have no result file yet (run_c.py / score_c.py untracked) |

## B. Positive but not chain-scored, or result pending

| item | section | evidence | blocker | code |
|---|---|---|---|---|
| tcnet base member | s274 | alone 45.1; marginal over {fb,fbl,sinc} +0.51 [+0.31,+0.71] 10/10 people, 4/4 cells; over the current four +0.22 (undecided); equal-n swap +0.63 [-0.12,+1.37] | Chain unscored. Base files exist only for 4 cells (within0/1 x draws 1/2, seeds 0-2): runs/diag/ttt/probs18/ttt18within{0,1}d{1,2}_tcnet_base-all_loso-s{0,1,2}.npz; the 8-order chain with 6 seeds needs new Leonardo training. s272 shows weak new members at weight 1.0 dilute the chain (eegn B -0.69), so it needs the s275 weight search form | studies/14nov/newmember/run.py ARM=A/B NEWSLOT=tcnet (family-agnostic; file naming must match) and run_w.py NEWW (s275) |
| s275 nested new-slot weights (eegn, conf; A_w / B_w) | s275 | pre-registered, no result file in evidence/newmember_15nov (only s272_score.txt); code run_w.py, score_w.py untracked | result pending (not read) | studies/14nov/newmember/run_w.py, score_w.py |
| s276 compound drops / halved seeds C1-C4 | s276 | pre-registered, no result file yet | result pending | studies/14nov/newmember/run_c.py, score_c.py, proxy_c.py, ret_c.py |
| Returning-person items (s247b two public sessions late fusion A: sealed +1.13; B: +0.12) | s247 | returning people only (P04 +4.31 / P10 +0.33 under A, P10 +/-0 under B); new people unchanged | not a new-people gain; needs a second public session, absent in the sealed stream | runs/strongpersonal |
| Run-split personal ceiling c x bE (+1.2 to +1.5 at w 0.5) | s252b / s265 / s267 | gain needs whole other runs of the session; delivery-prefix form V2: -1.23, S2 at w 0.5 +0.03 [-0.65,+0.67] 7/10 (runs scheme +1.14) | not deployable as measured; undecided at V2 w 0.5 | studies/14nov/personal_ceiling |

## Reading

Ready to stack in one shufcombo chain run (code exists, chain-scored mean > 0, no interval entirely below 0): GSCF (s251), EYEC copies (s235), ext/J (s239 part 1), and the s273 drops as time budget. All are below +0.5 each and none is in lean/src66, so a v6.7 candidate needs package code per item; the summed parts are about +0.3 to +0.6 against the s281 target of +1.0. The one item with a large measured piece, tcnet (+0.51), is not chain-scored and its files cover 4 of 8 orders.
