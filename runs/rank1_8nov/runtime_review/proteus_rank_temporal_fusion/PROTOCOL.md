# Rank covariance and bounded lag energy fusion

Use seven source participants and the frozen A/B/C participant folds.
Use auxiliary bootstrap seeds 2000, 2100, 2200 and 2300 with archived
overlap seed offsets 0, 100 and 200. Keep P04/P09/P10 outside fitting,
selection and scoring. The design uses only the source results from
3081129_0 and 3081432.

Fit separate heads for Hann real coherency plus bounded squared increments,
Gaussian-rank tangent covariance, and the matched ordinary tangent
covariance. The fused vote is the arithmetic mean of two independently
fit class posteriors, with no fitted weights. Fit all heads and covariance
references using the same bootstrap rows. Compare rank fusion with
ordinary covariance fusion and each single branch.

The primary comparison replaces the 10% power vote, holding its total
weight fixed. The secondary comparison adds the 10% auxiliary vote
to fixed 10%, matching the original covariance screen. Score both
using the official equal-cell balanced accuracy and require exact
source comparator parity. Advance the primary fusion only for +0.8 points
overall, +0.5 BrainHero, nonnegative Graz, +0.3 over every specified
control, and a positive gain in at least three of four auxiliary seed
averages. No automatic protected scoring or upload.
