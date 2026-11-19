# Lane B5 · Learning curve over the number of training people N, raw FBCNet family, standalone and in the cap250 full system

Registered Fri 20 Nov 2026 ~00:30 CET, before any posterior of this lane was scored. Code `laneB5/curve.py`, `laneB5/collect.py` (commit 215c265d + smoke fix).
Compute: Galvani a100-galvani (release `/mnt/lustre/work/gehler/gfh527/ni26_laneB5_20261120`). Ferranti not used: it has no repo/data checkout and
a separate filesystem, lane V had no setup notes yet (`laneV_virtual.md` absent), and the Lane H full-system consumer (oracle/outer/worker, pinned
Friday producers) lives on Galvani. Nothing about Ferranti is claimed.

## Design
Same net, optimiser, 40 epochs, SessionBN k 20, seeds 0-2 as Lane H arm A (`fbcnet_raw_sbn`, public PROTEUS, no trunk). Held person h in P01-P10 (all_loso);
training people = N of the other 9. N = 5, 6, 7, 8: S = 5 nested random subsets per (h, s) (first N of one fixed permutation, shared by seeds); N = 9: the full
LOPO set (equals Lane H arm A, re-run here as zero-change check). 25 configs x 10 held x 3 seeds = 750 fits, delivery order full0 only.
- Standalone: 3-seed ensemble BA full0, held-new, per config; per N mean over subsets; people-resampled 80 % intervals (cell-weighted, as the sealed set).
- Full system: for each config the bank replaces the five frozen raw-FBCNet seeds (laneH/full_replay.py, MASS 5/6, F-C20 form) in cap250 full0 replay;
  terminal BA per config, per N mean over subsets. Replay stock must equal the Friday c25d full0 (73.74), checked by `--cap250`.
Guards: collector refuses partial seed/person sets, wrong delivery order, rows without posteriors; curve.py asserts no trunk, 40 epochs, held not in training people.
Zero-change check: N9 standalone within +-0.5 of Lane H arm A (59.31 full0); N9 full-system within +-0.4 of the A-control (-0.15 vs stock).

## Predictions (BA %, held-new, 3-seed ensemble)
Standalone N = 5/6/7/8/9: 54.5 / 56.0 / 57.3 / 58.4 / 59.3; slope 7 -> 9 about +1.0 pp per person [+0.3, +1.8] (T-B small net: +1.1 to +1.9 per person).
Full system (5/6 swap): terminal minus N9 swap at N = 5/7: about -1.0 / -0.4 pp; slope per added person at N = 7-9 about +0.15 [-0.05, +0.4] (earlier standalone-to-stack
transfer about 0.1, and the member holds 5/6 of one family among many).
Extrapolation (log-linear fit on standalone, saturating power law as lower case): +5 people about +3 pp [+1.5, +4.5] standalone; full system about +0.4 pp, +10 about +0.7, +20 about +1.1.

## Rules (stated before scoring)
- Slope per person = (BA(9) - BA(7)) / 2 and from the fitted curve at N = 9; intervals resample people (cell-weighted), subsets averaged inside a person.
- Coverage lever is **sized in the full system** by the fitted +10-people gain: **material** if its people-80 % lower bound > +0.3 pp; **negligible** if the upper
  bound < +0.5 pp; otherwise **undecided**. Standalone steepness does not stand in for it. Bars are reading aids (instrument +-0.5 across people).
- Extrapolation range reports both fits (log-linear = optimistic, saturating power law with a free asymptote and exponent bounded in [0.2, 2] = conservative); the extrapolation is
  outside the measured N and says nothing about people of a different kind (lane V).
- Not covered: fb / fblight families, other orders, returning people, MASS other than 5/6, nets trained on external task data.

## Result (Fri 20 Nov ~01:30 CET; smoke 3138650, fits 3138654 (30 tasks, 750 fits, ~5 min each), full0 replays 3138787-3138807 (21 banks, ~4 min each); numbers `laneB5_curve.json`)
Zero-change checks passed: N9 standalone 59.31 = Lane H arm A exactly; N9 full system 73.59 = stock 73.74 minus 0.15 (the A control), `--cap250` stock check True.
BA (held-new, 3-seed ensemble, subsets averaged), N = 5/6/7/8/9:
| | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| standalone | 54.71 | 56.26 | 57.01 | 58.24 | 59.31 |
| full system (5/6 raw-family swap) | 73.98 | 73.91 | 73.73 | 73.69 | 73.59 |
Per person (people-resampled 80 %, cell-weighted):
| | standalone | full system |
|---|---|---|
| slope 7 -> 9 per person | +1.15 [+0.79, +1.57] | **-0.07 [-0.14, -0.00]** |
| gain 5 -> 9 | +4.60 [+3.36, +5.88] | -0.39 [-0.61, -0.18] |
| +5 people, log-linear / sat. power | +3.36 [+2.46, +4.28] / +3.03 [+1.86, +3.73] | -0.30 [-0.44, -0.16] / 0.00 |
| +10 people | +5.68 [+4.16, +7.25] / +4.97 [+2.87, +6.11] | -0.51 [-0.75, -0.27] / 0.00 |
| +20 people | +8.90 [+6.51, +11.35] / +7.47 [+3.94, +9.17] | -0.81 [-1.17, -0.43] / 0.00 |
Predictions: standalone levels within about 0.5 at N = 5 to 9 (54.7 vs 54.5, 56.3 vs 56.0, 57.0 vs 57.3, 58.2 vs 58.4); slope 7 -> 9 +1.15 vs +1.0 predicted, inside. Full system: predicted rise +0.15 per person; **it fell instead** (sign wrong, outside the interval).
Reading by the rules: coverage lever through this family in the full system is **negligible** (+10 people upper bound -0.27 log-linear, 0.00 saturating, both < +0.5). Standalone steepness did not carry (about 0 or slightly negative transfer).
Why is not settled: a weaker, less confident N = 5 member is not worse inside the chain (its errors are more complementary or its posteriors are better calibrated for the mix/EM stages); a stronger standalone net replaces that diversity. Untested here, so no claim. The negative slope is small (-0.07 per person, a bound at 0 on the high side) and inside the instrument's +-0.5 across people, so read it as "no gain", not as "more people hurt".
Not covered: fb / fblight families and the other members (the other 1/6 of each slot and all non-FBCNet members were fixed at 9 people), full0 only, MASS 5/6 only, returning people. For lane V: the effective N it must emulate for this family is irrelevant to the full-system score (flat 73.6 to 74.0 over N = 5 to 9); only standalone-level targets care about N.
