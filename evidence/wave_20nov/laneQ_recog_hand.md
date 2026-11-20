# Lane Q · cross-day returning recognition, handedness mirror, missing-class count (registered Fri 20 Nov 2026 ~10:05 CET)

Scoring order honesty: Test 3 (missing class) was computed BEFORE this card was written (local CPU, banks of `ni26_final_stack_20261120`,
`laneQ/missing_class.py`); its prediction below is post hoc. Tests 1 and 2 were not scored (smoke runs of the mirror fit and feature
extraction only; no gate score, no mirrored posterior was read).

## Test 1 · open-set recognition across days (stress test: MI-only corpora, other montage tasks, windows exported at 41 ch)
Instrument: the package IdentityGate (spectral identity; final_stack_hardcb.zip `bio_*` arrays, class `IdentityGate.scores` replicated
vectorised and checked against the class: zero diff) on the 1,599-d per-window feature (`laneQ/extract_feat.py`, checked equal to the class
`feat`), enrol ONE day/scene as a m=1 gallery row (as `niv/enrol.py` person_rows: mean feature, projected), stream the other days and other
people under random full shuffle (first 150 windows of each session, 4 orders), gallery = the 10 public rows + enrolled external people,
fixed rule `first n>=11 with max score > .26 latches` (ramp>0). NETBCI: 19 people x 4 days, enrol day 1, probe days 2-4 (true) and all days of
non-enrolled people (impostors), 40 random 50/50 people splits. ds007221g: 14 people, enrol scene 1 / probe scene 2 (and reverse; SAME DAY,
weaker). The contrast gate (.22) is NOT scored: it needs the shadow mixer posteriors of the full package and the corpora are 2-class (NETBCI) or
3 of 4 classes (ds007221g); say so wherever the numbers are quoted. So this is the spectral half of cap250's gate only.
Fix arms (all on the same scores): S-norm and AS-norm (top-25 cohort) with the cohort = sessions of the OTHER corpus (fixed vectors, no test
person), threshold chosen on the swapped split's impostors at FA <= stock FA; and "3 consecutive windows" persistence.
Predictions (true-recognition rate by 150 windows / median latency / impostor-session FA): stock NETBCI cross-day TA 55 % [30, 85], median age
~40, FA <= 3 %; ds007221g same-day TA 80 %. Fix: S-norm TA +5 points or more at FA <= stock: P = 35 %; AS-norm no better than S-norm.
Rule (Opus council, fixed now): adopt only if TA rises >= 5 points at FA <= stock FA on NETBCI AND ds007221g does not fall, AND the P04/P10
recognition age (visit 2, 3 against a visit-1-only gallery, spectral gate alone, 8 orders) does not get later AND held-new public people (gallery
without the person, 10 LOPO folds) keep zero latches. Otherwise: report, do not package. Intervals resample people (and the order draw).
Sealed cost model: 0.377 (8 one-session people) x miss rate x routed gain (12 pp P10-like to 40 pp P04-like, Lane R); upper bound, because the
contrast gate (OR) catches some of the misses.

## Test 2 · handedness mirror (diagnostic only)
Member: raw FBCNet SessionBN (laneH arm A recipe, `fbcnet_raw_sbn`) and fblight (`fblight_ea_rot05_sbn`), LOPO P01-P10, seeds 0-2,
orders full0-3: posterior on the held person for the original and the left-right mirrored session stream (homologous swap of 18 pairs, midline fixed;
mirroring before EA/SessionBN). `laneQ/mirror_lopo.py`, Galvani array 3140654. Views: orig, mirror, and mean of the two (extra view). Score:
per-person MI recall (class 0) and cell BA. Prediction: mirror alone costs 20-30 pp BA for every person (the net is trained unmirrored);
the averaged view changes BA by <= +-0.5 pp for most people; P05's MI recall improves by > 5 pp with P = 15 %, and no person improves MI recall
without a larger loss elsewhere. Supported only if the orig+mirror mean raises P05 MI recall >= 5 pp AND does not lower any other person's BA by
more than 0.5 pp AND the mean BA change is >= 0; else refuted for this form (a trained-in mirror augmentation is a different, unrun form).
The member is one stack member, not the full chain: a diagnostic, not a sealed estimate.

## Test 3 · missing class (10-minute check)
Count held-new cells (28 per order, full0 and within0) in which a class has <= 2 predicted windows, stock vs hard/cb/hardcb, stages raw, filtered,
terminal. Prediction (post hoc, see above): 0 of 28 in both orders for every arm.

## Results (Fri 20 Nov ~12:00 CET; feature extraction Leonardo 64925850; mirror fits Galvani array 3140654 (smoke 3140649); scoring local CPU; files `evidence/wave_20nov/laneQ/`, code `laneQ/`)
Guards: `check_feat.py` (feature == IdentityGate.feat, diff 0), `check_recog_lib.py` (vectorised scores == package class, 7.5e-15).

### Test 1 (spectral gate only; contrast gate not run, see registration). Mean over 40 person splits (E/U and swapped), 4 shuffles, gallery = enrolled people only (|E| 7-23)
"wrong" = share of probe sessions with a wrong-person latch (impostor latch + misID). Stock = tau .26, first check n=11.
| corpus (enrol -> probe) | stock TA / misID / impostor FA | median latch n | S-norm at stock wrong-rate | S-norm at 2 / 5 / 10 % wrong vs stock-threshold-raised at the same rate |
|---|---|---|---|---|
| NETBCI day 1 -> days 2-4 (cross-day) | .82 / .10 / .15 | 11 | .80 | TA .57/.66/.73 vs raw .34/.42/.55 |
| ds007221g scene 1 -> 2 and 2 -> 1 (same day) | 1.00 / 0 / .01 | 11 | .99 / .93 | raw already 1.00 up to 5 % wrong; S-norm .93-1.00 |
| ds007221lr ses 1 -> 2 (80 windows) | .78 / .02 / .19 | 11 | .63 | TA .10/.24/.57 vs raw .13/.33/.67 |
In-lab arm (`recog_proteus.py`, P04/P10 enrolled on visit 1 only, m=1, visits 2 and 3, 8 shuffles): stock spectral gate TA 8/8 in all four cells
(first latch ages P04|2 11-84, P04|3 11, P10|2 11-50, P10|3 11-29), 0 of 64 new-person streams latch (8 new people x 8 orders). S-norm at zero
impostor FA: TA 0/8 in every cell.
Reading: on genuinely cross-day foreign-lab data the stock gate does NOT miss many returns (TA .78-.82 by 150 windows, nearly all at the first
check) but OVER-latches (wrong-person latch about 13 % of sessions at tau .26, against about 0 % in lab); it is a threshold-calibration problem
of a foreign montage and lab, not a miss problem. The cohort score normalisation does not beat the stock threshold at the stock operating point
(NETBCI -2, ds007221g -1 to -7, lr -15 points TA at equal wrong-latch rate), helps NETBCI only at stricter rates (+19 to +24 points at 2-10 %) where lr
and ds007221g lose, and breaks the in-lab case (P04/P10 TA 0 at zero FA). Persistence (3 windows) changes nothing (latches are decided at n=11).
Registered rule (TA +5 at FA <= stock on NETBCI, ds007221g not lower, P04/P10 age not later, held-new zero latch): FAILS (NETBCI -2, ds007221g lower, P04/P10 TA 0).
**Refuted for this form (cohort = other-corpus sessions, S/AS-norm, 3-window persistence); no package built.** Not covered: an in-lab cohort
(only 14 public sessions exist), the contrast gate, an SPRT with calibrated per-person thresholds, a lab-matched calibration of tau.
Prediction check: stock NETBCI TA .55 predicted, .82 seen (high side), FA <= 3 % predicted, 15 % seen (wrong); ds007221g .80 predicted, 1.00 seen; fix +5 at stock FA (P .35): not met.
Sealed cost: miss rate (1-TA) is 0 in the only in-lab cross-visit evidence (32/32 streams) and 18-22 % on the stress corpora; upper bound
0.377 x 0.2 x (12..40 pp) = 0.9-3.0 pp if EVERY miss lost the routed gain with no contrast-gate rescue, which the in-lab 32/32 and the 100 % ds007221g
do not support; central estimate about 0 and an unmeasured tail. The fix would not recover it. The over-latch side (new people routed to a wrong head)
is the stress-corpus risk to watch; the in-lab 0/64 and the zero latches on held-new public LOPO (earlier lanes) say it does not occur on PROTEUS-lab data.

### Test 2 (mirror view, one member each, LOPO, 3 seeds, orders full0-3, cell-equal BA / MI recall %)
| member | person-mean BA orig / mirror / mean-view | mean-view delta BA | MI-recall delta mean | worst person BA delta | P05 MI recall orig -> mean |
|---|---|---|---|---|---|
| raw FBCNet SessionBN | 62.2 / 57.8 / 61.0 | -1.21 | -2.70 | -3.75 (P08) | 53.5 -> 40.2 (-13.3) |
| fblight_ea_rot05 | 57.9 / 51.7 / 57.1 | -0.82 | -0.87 | -6.51 (P06) | 52.0 -> 57.6 (+5.6) |
Only P03 (+2.0/+6.8 BA) and P09 (+3.3/+4.0 BA) improve under both members; P05 flips sign between members (-13.3 / +5.6 MI recall), so the P05
signal is noise. Rule (P05 MI +5 pp AND no other person below -0.5 AND mean >= 0): not met (mean negative, worst person -3.8 / -6.5). **Refuted for this form
(test-time mirror view, unmirrored-trained members)**; trained-in mirror augmentation not run. Prediction check: mirror-alone cost predicted 20-30 pp, seen 4.4-6.2 pp BA (the nets are
much less lateral than assumed; this is itself a weak sign the stock members barely use hemispheric asymmetry); P05 gain P = 15 %, not seen.

### Test 3 (missing class)
0 of 28 held-new cells with a class <= 2 predicted windows, in both full0 and within0, for stock, hard, cb and hardcb, at the raw, filtered and
terminal stage. Smallest class share in any cell: stock 13.5 % (full0 terminal), hardcb 14.6 %; within0 21.5 %. So FT/TTT and the filters do not drop a
class on the ten reused public people. Not covered: returning cells, Scherer, other orders. Prediction held (post hoc).
