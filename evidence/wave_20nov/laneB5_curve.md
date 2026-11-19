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
