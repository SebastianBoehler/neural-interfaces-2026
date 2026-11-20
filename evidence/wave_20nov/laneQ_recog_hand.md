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
