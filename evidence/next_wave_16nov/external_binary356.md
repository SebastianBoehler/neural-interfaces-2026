# §356 External cognitive-command evidence in the full dev chain

The qualified §352 experiment found that PROTEUS-trained 19-channel covariance features
transfer to Scherer WORD/SUB better than diagonal power. The reverse question matters
for the competition. Does a Scherer-trained specialist correct the incumbent's SUB/WORD
decisions after the existing chain has used its other evidence?

## Fixed intervention

Train the exact §352 binary covariance head with Scherer WORD/SUB labels only. Reuse the
19 shared native channels, 12 bands, three sub-window blocks, fold-independent per-external-
session centering, feature standardization, balanced logistic C=.05 and actual one-window
prediction. The 1,420 training rows have task labels independently confirmed against the
raw §351 manifest. All original external conditions enter the unlabeled session reference.
In PROTEUS each LOSO fold uses only the other nine people to form the k=20 causal
session-alignment prior. Preserve the original §352 PROTEUS-only LOSO covariance bank
as the matched additional-evidence control. The new producer never opens PROTEUS y.

Apply the terminal change after the latest count filter and causal personalization, using
the fully guarded §318 cap250 control. All recognition, fine-tuning, personalized features,
count filtering and pseudo-label states run the identical original paths in every arm.
For the terminal readout, preserve the incumbent decision whenever it is MI. Otherwise,
adjust its WORD-minus-SUB margin by weighted expert log odds, clipped to ±4 per expert.
Use the log probability ratio where the package returns probabilities, and the score
difference where it returns log scores. A zero weight pair returns exact original decisions.

| Arm | PROTEUS expert weight | Scherer expert weight |
| --- | ---: | ---: |
| Zero | 0 | 0 |
| Source | .15 | 0 |
| External | .15 | .10 |
| Reverse | .15 | −.10 |

This changes the terminal decision where the mechanism has a plausible semantic match.
§7/§191 pooled external data and §18/§132 used a three-way frozen transfer teacher.
§264 measured a different three-class cross-corpus contrast alone. §352 froze a
PROTEUS-trained pairwise model and ran the opposite transfer direction. None tests
the incremental contribution and signed control of a task-matched expert here.

## Predictions and limits

Use 10 PROTEUS people held out of all fold-specific fitting, eight actual delivered
orders and six dependent chain bases. Report official equal-cell and scheme-equal
people/order intervals, person effects including P04 and P10 as new people, and B2_pool
separately. Predict Source−Zero +0.10 [−0.25,+0.45], External−Source +0.10 [−0.20,+0.40]
and External−Reverse +0.20 [−0.25,+0.65] percentage points on the official metric.
External evidence must improve over Source and Zero in the same direction to pursue
this form. Upper interval below zero refutes the relevant fixed contrast. Effects
inside person noise or intervals including zero remain undecided. No return-cohort,
package runtime, sealed transfer, physiological-causality or rank estimate follows.

Before PROTEUS gold, require original §352 bank, masks and eight order pins, exact
external refit, finite active external logits, exact prefix and repeated-session
guards, original §318 complete provenance, zero-change, person/session/cell agreement
and exact unchanged MI decisions in all 48 cells. A terminal activity count of zero
means the registered contrast did not execute. The metric must reproduce the original
baseline. Source and data hashes are recorded before and after production.

One CPU2/8GiB/6min smoke with actual source and P01 inference only, no PROTEUS accuracy.
Release one CPU2/8GiB/25min full run only if the smoke verifies all guards and projects
under 20 minutes and 6 GiB. Maximum allocated 3,720 CPU-core-seconds, zero GPU.
No upload, new license acceptance, or frozen-package edits. Existing organizer
clearance and method disclosure requirement recorded in STATE applies. Scherer is
clinical, recorded with a different montage/reference and language, and its fixed
export includes 0.25 seconds of task cue. External results can reflect those differences.

## Complete result

The original smoke 3121985 stopped before PROTEUS labels. The newly introduced
sum-to-one tolerance was too strict for final probabilities cast from float32. Maximum
archived deviation was `4.47035e-8`. The observed probability rows exactly matched each
session's first 90 delivered windows across all eight pinned orders and six bases.
The revised smoke uses that exact boundary and the original §318 validator, with
the same model, odds and doses. It passed as 3121992 in 84 seconds.

Full allocation 3121998 completed in 8m48s. Replay printed OK. All 10 fold exclusions,
80 actual expert/order banks and 48 full-chain controls passed. Unchanged zero and
MI decisions were exact. The Scherer expert changed 3,070 decisions relative to the
PROTEUS-only expert across the dependent order/base comparisons.

| New-person full-system dev | Untouched | PROTEUS expert | + Scherer | Reversed Scherer |
| --- | ---: | ---: | ---: | ---: |
| Six-base official cell mean | 76.2177% | 75.7174% | 75.3670% | 75.6040% |
| B2_pool official cell mean | 76.2793% | 75.7429% | 75.3147% | 75.5768% |

The true Scherer direction lost **0.3504 percentage points** versus the PROTEUS-only
expert on the six-base official measure [80% people/order interval −0.5645, −0.1621].
The person-equal effect was −0.4117 [−0.6407, −0.1808], with 2/10 people up,
including +0.0055 on P04 and −0.3997 on P10 as new people. Its B2_pool official
effect was −0.4282 [−0.7156, −0.1839]. The standalone terminal PROTEUS correction
also lost 0.5003 points versus the untouched chain [−0.7590, −0.2730].
The true-versus-reversed contrast is undecided [−0.5843, +0.1035], so the test
does not establish a causal explanation for the degradation. The fixed terminal
fusion forms failed. The opposite-direction external discrimination result in §352
still holds at its binary scope.

Both smoke jobs and the full allocation used 1,402 CPU core-seconds and zero GPU.
Full reported step peak was 2,716,556 KiB. End-to-end runtime inside a competition
worker was outside this study. [Before-label proof](external_binary356_before_labels.json)
and [full results](external_binary356_result.json) retain source hashes,
all arms, both weightings and P04/P10 effects.
