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
