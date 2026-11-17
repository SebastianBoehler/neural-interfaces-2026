# §357 Unordered-prefix readout readiness

§307/316 learned update scaling for a tangent-core head. §323/332 adapted session
normalization, and §346 changed the readout from current-window features. None of
these executable paths trains the decoder to use a pooled set of prior windows.
§346 saved exact by-order causal A6 probabilities but did not retain cross-fitted
support embeddings. This probe qualifies a minimal source path before any accuracy
experiment. [Prior coverage review](exploration_primary_sources_17nov.md)

Use the frozen **A6 raw FBCNet/SessionBN** body trained on P04/P05/P06/P07/P08/P10.
Fit a residual readout offline on P02 only and run unlabeled readiness guards on
P03. Both people are excluded from the body fit. Neural forwards use no P01 or
P09 windows and the probe never opens P03 or P09 gold. The owning loader still
materializes metadata/raw arrays and its existing per-window covariance cache.
The path retains A6's exact cached causal current probabilities for all eight
pinned orders, with an independent within0 rerun and original error bound.

Extract each 1,152-dimensional support item with the same body in B1 mode using
its fixed population-normalization statistics. This makes support items
independent of their arrival order. Both readouts learn a 16-dimensional tanh
transformation per support item, followed by a 16-unit residual head conditioned
on the original three current log probabilities. The residual is bounded by
±0.25 per class logit. Identical seeds, full source batches, capacity and 16
Adam steps. The current-only control takes only the transformed current support
item. The prefix arm takes the mean of transformed **previously delivered**
items, with zero context at empty prefix. The existing causal A6 prediction
remains order-dependent. Invariance refers to the new support-set contribution
for the same multiset with the same current base prediction.

On all eight actual P03 orders, check step-by-step versus vectorized reference
and fixed-multiset reversal to 1e−12, exact prefix and reset/interleaving identity,
exact disabled A6 scores and no use of current-item data in its own support set.
Predict support gradients above 1e−8 and at least 1e−6 prediction sensitivity to
changing the actual prior items. Any failed guard blocks readiness. Owing to only
one readout training person, no accuracy is scored or inferred from offline loss.

Fixed one CPU2/8GiB job capped at eight minutes (960 allocated core-seconds).
No training body updates, GPU, official scoring, further query release or upload.

## Readiness passed

Job 3122027 completed in 52 seconds with two CPUs (104 allocated CPU
core-seconds, zero GPU). The Python process took 23.87 seconds and reported
2,265,464 KiB peak RSS. Both matched readouts have 18,819 parameters. The
source gradients reached 0.01098 for the current-only arm and 0.01080 for the
prefix arm. The independent A6 rerun retained all 378 predicted labels and
had at most `5.61e-7` probability deviation from the pinned original banks.

All eight unlabeled P03 orders passed. Vectorized training and streaming
replay agreed within `2.22e-16`. Reordering the same prior multiset changed
the output by at most `5.55e-17` at the fixed current prediction. Reset,
prefix isolation, disabled readout, strict current-item exclusion and
interleaving with P02 passed exactly. Changing the prior items changed one
measured probability by 0.05188, confirming that the support contribution ran.

This qualifies the support representation and causal readout. It used one
readout-training person and read no P03 or P09 labels. There is no held-person
accuracy or system benefit estimate. No query run was released.
[Run proof](causal_prefix357_proof.json)
