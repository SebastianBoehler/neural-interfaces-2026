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

## §358 Four-person input qualification

The same A6 body excludes P01, P02, P03 and P09. Archived A6 current probabilities
cover eight orders each on P02/P03 (§346 extension) and P09 (§338). P01 has no
matching archived current bank. §357 checked the population-normalized B1
embedding path on P02/P03 and kept hashes only.

One label-free CPU2/8GiB/8min allocation builds all four people afresh on the
same route. Restore the checkpoint with the owning FBCNet builder, independently
recompute source-only G/scale, and never call the training preparation path or
load any labels. Each person needs one current-window support feature per row
and fresh complete causal current probabilities for all eight orders. Strictly
check support order independence, prefix and repeated-session isolation, original
row identities, and all archived current banks to `1e-6` with zero changed labels.
Predict worst archive error below `1e-6` based on §357's `5.61e-7` single-order
measurement. Any failed guard blocks the instrument for scoring. This releases
only the bank build.

The smallest subsequent accuracy test is four leave-one-person-out folds with A6
fixed. Fit the same 18,819-parameter current-only and prefix readouts with the
same seed, dose, 16 optimizer steps, and eight training orders on the other
three A6-excluded people. Compare both complete decoders to untouched A6 on every
pinned order. Require exact disabled identity and prefix/reset checks, active
gradients and actual support sensitivity, complete folds before held scoring, and
no use of held labels by that fold's fit. Use the official cell-averaged metric
and people/order intervals. Four exposed people can size effects for a later
experiment, with limited independent evidence.

A proposed follow-up budget is one CPU2/6min fit allocation and one CPU1/3min
collector after measuring the bank. This would be a separate registration.
The input build itself has zero optimizer steps and opens no labels.

Job 3122038 completed in 2m01s on two CPUs. It built all 757 support items and
32 causal current banks. The 24 archived checks retained every predicted label
and reached at most `6.10e-7` in probability error. P01's eight new banks passed
the same prefix, repeated-session and order checks. Across all four people the
largest early-prefix error was `5.44e-7`. The independent source G/scale and frozen
model guards passed. The Python process took 96.84 seconds and reported 2,019,816
KiB peak RSS. [Bank proof](causal_prefix358_proof.json)

The earlier job 3122033 stopped at a malformed archive context after the label-free
restore and before any cohort forward. The repaired allocation used the same model,
thresholds, cohort and bounds. Total §357–358 allocated compute was 402 CPU core-seconds.
The verified 3.2 MiB bank stays in Galvani at
`ni26_prefix358_v2_20261117/banks.npz` with SHA-256 `92770628…db5a`.
Neither bank allocation read labels, updated parameters or computed accuracy.
