# §359 Four-person held pilot for the unordered-prefix readout

Use the four label-free §358 A6 inputs without changing the body, probabilities,
support features, rows or orders. In each fold, one of P01/P02/P03/P09 is held
out of every readout fit. The other three provide source labels for the same
16-step full-batch Adam recipe as §357. Source losses use class balancing.
There is no feature fitting, temperature calibration or checkpoint selection
on the held person. Both heads use 18,819 parameters, the same seed and a
fixed ±0.25 logit residual. One sees a transformed current support item and
one sees the mean of transformed strictly earlier items. The untouched
causal A6 probabilities are the third, strongest reference.

An actual P02 source fit checks the generalized implementation against the
original single-person routine, with identical parameters required. This
separate plumbing check opens no accuracy labels or scores. In each of the
four subsequent folds, only that fold's three fit people supply training
labels. The owning label reader explicitly receives those row IDs.

Before using labels in the scoring role, require all 96 posterior banks and
their source proofs. Check equal initialization/capacity, active gradients
and decisions, prefix invariance, current-item exclusion, reset, interleaving
and exact disabled scores. Interleave the shared 189 actual rows in the
one unequal-length case. Verify the vectorized prediction and streaming
readout agree within `1e-12` on every complete stream, including P01's extra
row. No matrix or held metric may be partial.

Use the original official cell-averaged balanced-accuracy implementation and
all eight orders. Average within/full/runs equally and report paired
people/order 80% intervals with nested resampling, plus each person's effect.
Predict Prefix−Current +0.25 pp [−1.50,+2.50] and Prefix−A6 +0.10 pp
[−2.00,+2.00]. A negative upper interval refutes a fixed contrast. A small
effect or interval crossing zero is inconclusive. The experiment uses four
previously exposed people, one fixed A6 checkpoint, and correlated folds
with overlapping source sets. The resampling describes that pilot and does
not supply independent confirmation.

One CPU2/8GiB/5min allocation includes the source smoke, all four fits,
eight-order prediction and scoring. No whole-system integration, package
change, alternate dose, retry by result, or upload.

## Result

Job 3122050 completed in **47 seconds** on two CPUs (94 allocated CPU
core-seconds, zero GPU). The source recipe smoke matched the original
parameters bitwise. All 96 banks passed before labels entered the score.
The prefix head changed 14, 51, 29 and 63 decisions versus the current-only
head for P01, P02, P03 and P09 across the dependent eight orders. Prefix,
reordering of a fixed prior set, current-item exclusion, disabled identity,
and interleaved session state passed.

| Fixed A6 whole decoder | Official cell BA | P01 | P02 | P03 | P09 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untouched A6 | 60.785% | 57.206% | 65.646% | 56.228% | 64.061% |
| Matched current-only head | 59.187% | 56.078% | 64.133% | 56.712% | 59.825% |
| Unordered-prefix head | 59.676% | 56.341% | 66.034% | 56.547% | 59.782% |

Prefix minus current-only was **+0.489 pp** [80% nested people/order interval
−0.052, +1.037], with gains on two of four people. Prefix minus untouched A6
was **−1.109 pp** [−2.296, +0.088], including −4.279 on P09. Both prefix
contrasts remain undecided under the registered rule. The matched current-only
head lost **1.598 pp** versus A6 [−2.767, −0.460]. The pilot gives no basis for
replacing A6. Equal-cell and person-equal estimates coincide because all four
people have the same number of scored cells.

This is a whole-readout, exposed four-person comparison, with overlapping
training folds and one body checkpoint. The study used 496 allocated CPU
core-seconds through support qualification and scoring. [Before-label proof](causal_prefix359_before_labels.json)
· [Full results](causal_prefix359_result.json)
