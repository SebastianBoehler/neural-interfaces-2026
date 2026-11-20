# Lane T3 · TTT-able base candidates (team C interface), held-new standalone with and without test-time training

Registered Fri 20 Nov 2026 ~08:45 CET (Leonardo AIFAC_P02_1636, release `/leonardo_work/AIFAC_P02_1636/ni26_tttbase_20261120`),
before any lane T3 posterior was scored. Read first: CLEANSLATE.md, cleanslate_teamC.md (A0 56 / A1 +1.5 / A2 +1.5 predictions;
only P07/P09 partial logs seen there: A1 73.0/68.8, A2 78.3/76.7, A2h 81.5/77.3), laneH (A 59.31, B 59.81), laneB6 card, teamA table.
Code `cleanslate/ttt_base/` (`ref/` = team C files vendored from 819765ac, `train.py`, `archs.py`, `merge.py`, `leo.sh`).

## Hypothesis
TTT gains add on top of the base, so the strongest TTT-able base wins. "Gain" = with-TTT minus no-TTT on the SAME base.

## Instrument (identical for every base)
Held-new LOPO: fold P trains on the other nine people (all sessions), P's sessions streamed through the package's own
`TTTModel.predict`, one window per call, full-shuffle orders 0 and 1 (`runs/shuffle_perms.npz` sha 6046bf5b, `full|k|all_loso`),
fresh model per session, official cell-mean BA (28 cells). Team C's `stream_eval` variants, on identical packages:
**no TTT = A1** (causal EA + prefix BN; A0 = source EA only, reported), **with team C's default TTT = A2** (4 steps, lr 1e-4)
and **A2h** (8 steps, lr 3e-4, team C's heavier form, the strongest registered one so far). ms/window = mean per-`predict`
seconds on one Leonardo A100 (x2.5 for the RTX PRO 4500 as team C assumes, budget 179 ms); it grows with ensemble size.
Intervals: people-resampled 80 % (cell-weighted) and paired people+orders 80 % for differences (10k draws). Dev weighting like
the sealed set (about 57 % new people) is an UNVERIFIED assumption; held-new only is reported, returning is not scored here.

## Bases (members trained with team C's loop: causal per-epoch EA, session-homogeneous batches, ls 0.1, AdamW one-cycle, rot 0)
- **a0** control E6: shallow/eegnet/fbnet x seeds 0,1, 60 epochs (team C's base re-run in this lane).
- **a1** E6 with seeds 0-3 (12 members), 100 epochs.
- **b0** FBCNet (braindecode, same net as Lane H) from scratch x seeds 0-2, 40 epochs (control for the trunk).
- **b1** FBCNet initialised from the Lane H arm-B trunk `hp7` (7 corpora) x seeds 0-2, 40 epochs.
- **c1** FBCNet initialised from `hbx` (hp7 trunk + ds007221 Graz 14 + Graz-paradigm 48 + NETBCI 19 people, B6 arm X) x seeds 0-2, 40 epochs.
- **d1** wider/deeper nets from scratch: FBCNet n_filters_spat 64, FBCNet stride_factor 8, FBNet m=8 (one seed each pair, see table), EEGNeX x2, IFNet x2; each also scored standalone as its own base.
- **e** MANAS-2 end-to-end (team A, selected B6 A1 fit) as an ensemble member: see "Not run / limits" until scored.
- **mix** (after the above): union of the best members (`merge.py`, eval only). Selection overlaps with evaluation; reported as such.
Silent-no-op guards: `train.py` asserts the trunk loads (all but 4 keys), generated `submission.py` asserts 3 BatchNorm sites patched,
`stream_eval` asserts the held person is not in the package's people; `train.json`/`weights.pt` list member name, kw, trunk keys moved.

## Predictions (BA %, held-new, mean of orders 0/1; 80 % people-resampled intervals)
A1 levels: a0 57 [51, 63]; b0 56 [50, 62]; b1 56.5 [50, 63]; c1 56.5 [50, 63]; EEGNeX / IFNet each 52 [44, 60].
TTT gain A2h - A1: +4 [0, +8] on a0 (team C's two-person look is +8, treated as optimistic); other bases within +-1.5 of a0's gain.
Differences (paired people+orders 80 %): a1 - a0 A1 +1.0 [-1, +3]; b1 - b0 A1 +0.5 [-1.5, +2.5] (Lane H +0.5); c1 - b1 +0.3 [-1.5, +2];
wide FBCNet - b0 +0.5 [-2, +3]. Best base with TTT minus a0 with TTT: +2 [-1, +5]. Gains are expected NOT to depend on base strength by more than 1 pp.
## Rules (stated before scoring)
Per comparison, paired 80 % interval of the difference: supported if the lower bound > 0 on the order-mean, refuted if the upper bound < 0, else undecided.
Instrument noise about +-0.5; bars are reading aids. Hypothesis "gain adds on top": supported if the A2h - A1 gain of every base lies within
+-1.5 pp of a0's (no shrinkage) AND the best-A1 base is also best with TTT; refuted if the gain falls by more than 2 pp on the strongest A1 bases.
"Best base" for team C = highest A2h BA among bases whose A100 ms/window x 2.5 <= 179; reported with its interval, selection among about 8 bases is stated as selection.
Not covered: returning people, other TTT forms than A2/A2h, other epochs/dose for trunks, orders other than 0/1.

## Addendum 1 · registered ~08:30 CET after seeing only the a0 and b0 summaries (b1, c1, d1, a1 unscored)
Seen: a0 A0/A1/A2/A2h = 60.3 / 63.1 / 65.0 / 62.1 (A2 - A1 +1.8 [+0.9, +2.9], A2h - A1 -1.1 [-2.7, +1.2]); b0 (braindecode FBCNet) 57.6 / 62.1 / 57.4 / 53.2
(A2 - A1 -4.7 [-6.6, -2.5], A2h - A1 -8.9). So the gain does NOT carry over to every base for team C's A2/A2h form (the "gains add" clause is already refuted for b0 and
this form); the level predictions above were too low (a0 A1 63.1 vs 57). Question now: is the FBCNet loss a TTT-form problem (drift) rather than a base problem?
Two gentler forms added to the vendored `ref/stream_eval.py` (labelled ttt_base, not team C's): **A2l** 4 steps lr 2e-5 all params; **A2bn** 4 steps lr 1e-3 BN affine only
(`params=bn`). Eval-only on the existing packages (b0, b1, c1 and the d bases), orders 0/1. Prediction on FBCNet bases: A2l - A1 -0.5 [-3, +1.5]; A2bn - A1 +0.3 [-2, +2];
neither beats A1 by a lower bound > 0 (supported only if so, refuted if both upper bounds < 0, else undecided). a0 is evaluated with the same forms as the shallow/EEGNet/FBNet reference.

## Addendum 2 · registered ~08:50 CET after the first full table (a0, b0, b1, c1, d_fbc64, d_fbcst8, d_fbwide, d_eegnex, d_ifnet scored; a1 and MANAS not)
Seen (A1 / A2 / A2h, mean of orders 0,1): a0 63.1/65.0/62.1; b0 62.1/57.4/53.2; b1 62.2/58.7/53.0; c1 62.4/58.9/52.1; fbc64 61.6/57.9/54.2; fbcst8 61.3/57.2/53.9; fbwide 58.4/59.1/56.3;
IFNet 56.9/57.6/53.2; EEGNeX 53.0/48.9/42.8. Trunk effect on A1: b1-b0 +0.07 [-0.6, +0.7], c1-b1 +0.18 [-0.4, +0.6] (undecided). The braindecode nets lose 3.5-10 pp under TTT; a0 gains +1.8.
Question registered now: which a0 member family carries the TTT gain (and which one loses)? Bases e_shallow, e_eegnet, e_fbnet = three seeds of one family each (60 epochs, same recipe), variants A0, A1, A2, A2h, A2l, A2bn.
Prediction: gain A2 - A1: shallow +1.5 [-1, +4]; eegnet +1.5 [-1, +4]; fbnet +1 [-2, +4] (team C's own FBNet shares their FIR-initialised filter bank); every one of the three above the braindecode nets' gain.
If the three gain and the braindecode nets lose, the working explanation is the parametrisation (max-norm / constrained layers, log-variance heads), not test-time training itself; this will be stated as an explanation, not a result.
Next whole-ensemble question (after these): do the braindecode members help a0 inside the ensemble under TTT? Union packages by `merge.py` (a0 + c1, a0 + d_ifnet), variants A0, A1, A2, A2l, reported against a0 on the same people/orders.

## Addendum 3 · registered ~09:25 CET (m_a0c1 and m_a0ifnet scored-in-progress, not looked at; m_sf, m_sfc, m_sfi submitted, not looked at)
Seen before this: e_shallow / e_eegnet / e_fbnet single-family bases (A1 61.9 / 49.2 / 58.9; A2 gain +0.8 / 0.0 / +1.6; A2l gain +2.2 / 0.0 / +0.8). EEGNet (team C's own definition) is near chance alone (49.2), so a0 may carry a
dead member; MANAS late fusion (member e, form added post hoc): a0 A2 + 0.25 MANAS 66.35 vs A2 64.95 (paired +1.39 [+0.23, +2.52]; weight grid 0.1-0.4 chosen after seeing it, plateau 0.1-0.25, w=0.5 loses; MANAS alone 58.68 on these orders, team A 58.79 on orders 71000-7).
Ensembles by union of existing members (no retraining), TTT teacher = the whole union (all members updated by A2): m_a0c1 = a0 + c1, m_a0ifnet = a0 + d_ifnet, m_sf = e_shallow + e_fbnet (no EEGNet), m_sfc = m_sf + c1, m_sfi = m_sf + d_ifnet. Variants A0, A1, A2, A2l.
Predictions vs a0 on the same variant (paired people 80 %): m_a0c1 A2 -0.5 [-2, +1]; m_a0ifnet A2 0 [-1.5, +1.5]; m_sf A1 +0.5 [-1.5, +2.5], A2 +1.0 [-1, +3]; m_sfc A2 +0.5 [-1.5, +2.5]; m_sfi A2 +0.5 [-1.5, +2.5].
Reading rule as above (lower bound > 0 supported, upper < 0 refuted, else undecided). Selection among these ensembles is selection on the 10 dev people and is reported as such.

## Addendum 4 · registered ~10:00 CET, after the union ensembles were scored (a0+c1 A1 65.47 / A2 66.28; shallow+fbnet A1 63.30 / A2 65.76; +c1 64.81 / 65.45; a0+ifnet 63.48 / 64.68; all with the whole union as TTT teacher and learner)
Reading so far: braindecode members lift the A1 level of the union (a0+c1 minus a0, A1: +2.33 [+1.09, +3.66]) but their TTT gain is negative or zero alone and drags the union's gain (+0.8 vs a0's +1.8). Hypothesis: freeze them during the
gradient steps (BN refresh only) and TTT only team C's own members. Implemented as TTT key `freeze_bd` in the generated package code (`train.py`, one patched line in `_steps`; `merge.py` with `REGEN=1` rewrites the code of existing packages, weights unchanged).
Variants **A2f** (A2 with freeze_bd=1) and **A2lf** (lr 2e-5, freeze_bd=1). Bases: **m_a0c1f** = a0 + c1 (must reproduce A1 65.47 exactly: no-op guard on the regenerated code) and **m_a0fbc** = a0 + c1 + d_fbc64 + d_fbcst8 (15 members, 9 braindecode).
Predictions: m_a0c1f A2f - m_a0c1 A2: +0.7 [-0.8, +2.2]; A2f - a0 A2: +1.5 [0, +3]; m_a0fbc A1 - m_a0c1 A1: +0.5 [-1, +2]; m_a0fbc A2f - m_a0c1f A2f: +0.3 [-1.2, +1.8]. ms/window: m_a0c1f A2f below m_a0c1 A2 (107 ms) and
near a0's 63 ms + forward cost. Same reading rule. This is a form chosen after seeing the unfrozen result, so it is reported as selected, not as a clean test.

## Addendum 5 · ~10:15 CET
Leonardo SSH certificate expired (publickey denied, ~10:05 CET); not renewed by this lane. Leonardo jobs 64926006 (m_a0c1f) and 64926007 (m_a0fbc) of Addendum 4 were queued and cannot be collected until access returns.
The freeze_bd question is duplicated on Galvani A100 (release `/mnt/lustre/work/gehler/gfh527/ni26_tttbase_20261120`, same code, new training run): a0g (E6, retrained), b0g (scratch FBCNet x3, no trunk: hp7/hbx trunks sit on Leonardo only; c1 vs b1 vs b0 differ by <0.3 at A1),
m_a0b0g = a0g + b0g with variants A1, A2, A2f, A2lf (jobs 3140867 a0g, 3140868 b0g, 3140869 mix). Predictions as in Addendum 4 with b0g in place of c1: A2f - A2 (same union) +0.7 [-0.8, +2.2]; A2f - a0g A2 +1.5 [0, +3]. Numbers from a retrained a0 differ from a0 (run-to-run); comparisons are within the Galvani set.

## Results (Fri 20 Nov, ~11:45 CET; held-new LOPO P01-P10, orders 0 and 1, official cell mean, 28 cells; people-resampled 80 % in brackets)
Leonardo A100 (AIFAC_P02_1636): arrays a0 64917099, a1 ..100, b0 ..102, b1 ..103, c1 ..104, d_fbc64 ..105, d_fbcst8 ..106, d_fbwide ..108, d_eegnex ..111, d_ifnet ..125, e_shallow/eegnet/fbnet 64920267-69,
gentle-form evals 64919255/57, 64920193-211; unions 64922273 (a0+c1), 64922275 (a0+ifnet), 64923292/94/99 (m_sf, m_sfc, m_sfi), 64926006/07 (m_a0c1f, m_a0fbc, REGEN code, A1/A2 reproduced exactly);
MANAS late-fusion streams 64917392 + 64919690; all-ten-people deployment package 64930202. Galvani A100 replication 3140867-69. Code revision: commit with this file (cleanslate/ttt_base/). ms/window = mean predict() seconds on one A100 (x2.5 = RTX PRO 4500 estimate, budget 179 ms => A100 <= 71 ms).

| base (members) | A0 | A1 no TTT | A2 TTT (team C default) | gain A2-A1 | A2h | A2l / A2bn gain | ms A1 / A2 |
|---|---|---|---|---|---|---|---|
| a0 E6 shallow/eegnet/fbnet x2 seeds, 60 ep (control) | 60.3 | 63.1 [58.0, 70.4] | **65.0 [59.9, 72.2]** | +1.8 [+0.9, +2.9] | 62.1 (-1.1) | +1.0 / +0.6 | 9 / 63 |
| a1 same x4 seeds, 100 ep | 61.1 | 62.7 | 64.1 | +1.4 [+0.4, +2.8] | 62.0 | | 15 / 120 |
| b0 braindecode FBCNet x3 scratch | 57.6 | 62.1 | 57.4 | -4.7 [-6.6, -2.5] | 53.2 | -0.7 / -0.1 | 20 / 43 |
| b1 FBCNet from Lane H hp7 trunk | 58.3 | 62.2 | 58.7 | -3.5 | 53.0 | +0.1 / +0.1 | 20 / 43 |
| c1 FBCNet from B6 hbx trunk (+81 MI people) | 58.8 | 62.4 | 58.9 | -3.5 | 52.1 | +0.2 / +0.1 | 20 / 43 |
| FBCNet 64 spatial filters / stride 8 | 58.1 / 57.9 | 61.6 / 61.3 | 57.9 / 57.2 | -3.7 / -4.1 | 54.2 / 53.9 | -0.3 / +0.8 ; -0.5 / -0.1 | 21 / 48 |
| IFNet x3 / EEGNeX x3 / FBNet m=8 x3 | 52.2 / 51.8 / 56.1 | 56.9 / 53.0 / 58.4 | 57.6 / 48.9 / 59.1 | +0.7 / -4.1 / +0.7 | 53.2 / 42.8 / 56.3 | +0.6 ; +0.7 ; +0.9 | 10 / 34 ; 12 / 117 ; 4 / 37 |
| e_shallow / e_eegnet / e_fbnet x3 each | 58.3 / 48.7 / 55.3 | 61.9 / 49.2 / 58.9 | 62.7 / 49.2 / 60.5 | +0.8 / 0.0 / +1.6 | 59.9 / 45.4 / 57.6 | +2.2 ; 0.0 ; +0.8 | 5 / 38 ; 7 / 40 ; 4 / 28 |
| m_a0c1 union (a0 + c1, TTT on all 9) | 62.0 | 65.5 | 66.3 [60.9, 74.0] | +0.8 [0.0, +1.9] | | 66.1 (A2l) | 28 / 107 |
| **m_a0c1f** same, `freeze_bd` (A2f) | | 65.5 | **66.7 [61.4, 74.2]** | +1.3 [+0.7, +1.9] | | 66.1 (A2lf) | 46 / **71** |
| m_a0fbc (a0 + c1 + fbc64 + fbcst8, A2f) | | 65.3 | 65.7 | +0.4 | | | 71 / 133 |
| m_sf (shallow + fbnet x3, no EEGNet) | | 63.3 | 65.8 | +2.5 [+1.6, +3.5] | | | 8 / 89 |
| m_a0ifnet / m_sfc / m_sfi | | 63.5 / 64.8 / n.s. | 64.7 / 65.5 / -0.8 vs a0 | +1.2 / +0.6 / | | | |
| MANAS B6-A1 (team A) alone, my orders | | 58.7 | | | | | ~11 (team A, A100) |
| a0 A2 + 0.15-0.25 MANAS posterior | | 64.5 (A1 + 0.25) | 66.3 [60.9, 73.7] | | | | |
| m_a0c1f A2f + 0.10 MANAS | | | **67.5 [62.3, 74.6]** | | | | |

Paired differences (people-resampled 80 %): a1 - a0 A1 -0.42 [-1.11, +0.42], A2 -0.83 [-1.86, +0.62] (undecided; double the run time); b1 - b0 A1 +0.07 [-0.56, +0.72]; c1 - b1 A1 +0.18 [-0.42, +0.62] (trunks undecided, as Lane H/B6);
m_a0c1 - a0: A1 +2.33 [+1.09, +3.66], A2 +1.32 [+0.11, +2.84]; **m_a0c1f A2f - a0 A2 +1.78 [+0.36, +3.25]**, A2f - m_a0c1 A2 +0.46 [-0.51, +1.37]; m_sf - a0 A2 +0.81 [-1.31, +2.56]; m_a0ifnet - a0 A2 -0.27 [-1.18, +0.86];
m_a0fbc A2f - m_a0c1f A2f -1.03 [-1.90, -0.34] (more braindecode members hurt); MANAS fusion: a0 A2 + 0.25 M - a0 A2 +1.39 [+0.23, +2.52]; m_a0c1f A2f + 0.10 M - A2f +0.76 [+0.26, +1.10], - a0 A2 +2.54 [+1.02, +3.93].
Galvani replication (retrained a0g 62.6 / 64.6, scratch FBCNet b0g 62.1, no trunk): m_a0b0g A1 65.3 (+2.75 [+1.56, +4.09] over a0g), A2 65.5, A2f 65.6 (A2f - A2 +0.06 [-0.95, +0.92]; A2f - a0g A2 +0.99 [-0.19, +2.22]).

Reading against the rules. **"Gains add on top of the base, strongest base wins": refuted as stated.** A1 level does not predict the with-TTT level: the FBCNet/EEGNeX family reaches A1 61-62 but loses 3.5-10 pp under team C's A2/A2h
(and ~0 under gentler forms), refuting the gain-within-+-1.5 clause for every braindecode base; only team C's own shallow / FBNet nets (and braindecode IFNet at +0.7) gain. The one clause that survives: the best base with TTT is the union that is also best at A1 (m_a0c1: A1 65.5, A2 66.3).
Predictions that missed by more than their interval: a0 A1 63.1 vs 57 predicted; b/c trunk A1 effects fine (+0.1, +0.2 vs +0.5, +0.3); A2h - A1 +4 predicted, -1.1 seen (team C's two-person look of +8 did not hold over ten people); unions m_a0c1 A2 +1.3 vs -0.5 predicted.
**A2h (8 steps, lr 3e-4) is worse than A2 on every base here** (a0: 62.1 vs 65.0) and its ms (105) is over budget; A2 is the useful form on the 10 dev people.
Not tested / limits: orders 0,1 only; returning people not scored; C is a 2-order, 10-person dev measure with selection among about 15 bases and several post-hoc forms (A2l, A2bn, A2f, MANAS weights), so the +1.8 and +2.5 are selected numbers; trunk arms c1/b1 have no Galvani replication (trunks live on Leonardo);
MANAS enters only as a late posterior mix (the TTT teacher does not see it); EEGNet (team C's own) alone is ~49 % (chance-level next to the 62 of the others) yet a0 minus EEGNet is not better (m_sf - a0 A2 +0.8 [-1.3, +2.6], undecided).

## For team C
Best TTT-able base under the budget: **m_a0c1f = team C's E6 (shallow/eegnet/fbnet x seeds 0,1, 60 ep) plus three braindecode FBCNet members initialised from the B6 `hbx` trunk (40 ep), with `freeze_bd=1`**
(braindecode members get the prefix BN refresh and enter the ensemble posterior/teacher, but take no gradient steps; team C's own nets run A2: 4 steps, lr 1e-4): held-new 66.7 [61.4, 74.2] vs 65.0 for E6 + A2, paired +1.78 [+0.36, +3.25]; 71 ms/window on A100 (x2.5 = 178 ms, at the 179 ms line, so no margin).
Without freeze: 66.3 at 107 ms (over budget). Adding `+ 0.10 x` MANAS posterior (fold packages: Leonardo `/leonardo_work/AIFAC_P02_1675/ni26_cleanslate_A2/b6-a1-manas/new_Pxx`, deployment ZIP in `b6-a1-manas/teamA-manas.zip`): 67.5 [62.3, 74.6], +2.5 over E6 + A2; late fusion needs MANAS posteriors from its Solver (patched one-line tap, `manas_stream.py`).
Plain E6 + A2 (65.0, 63 ms) stays the fallback; more seeds/epochs (a1) did not help (-0.8, 120 ms).
Where: code `cleanslate/ttt_base/` (`submission.py` of a package = `ref/submission.py` + `archs.py` + generalised BatchNorm + `freeze_bd`; arch `bd` = braindecode net; copy that file with `weights.pt` to run team C's `stream_eval` variants unchanged; new variants A2f/A2lf/A2l/A2bn in `ref/stream_eval.py`).
Weights (Leonardo, `/leonardo_work/AIFAC_P02_1636/ni26_tttbase_20261120/`): held-new fold packages `work/m_a0c1f/pkg_P01..P10` (trained on the other nine); **all-ten-people deployment package `work/final10/pkg`** (weights.pt 895 KB, sha256 0bd39c34...; submission.py sha256 a24468da...; hbx trunk inputs `runs/trunks/`, from Lane B6 `ni26_laneB6_20261120/runs/pretrain`, corpora and licences as in the B6 card). Nothing committed; weights stay on the cluster. The all-ten package has not been streamed through the shared harness (no worker-image timing, no reset/interleave guard beyond team C's own `stream_eval` protocol); do that before any upload claim.
