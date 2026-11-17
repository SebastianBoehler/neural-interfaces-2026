# §352 Frozen PROTEUS-only WORD/SUB external stress test

Status: **running**, one bounded Galvani CPU allocation. This is a binary external condition-discrimination
study, not a three-class, current-system, sealed-score or physiological-causality result.
Root authorized this single test after §351 qualified all raw trial/group metadata.
Actual source, class, grouping, model, numeric predictions and resource limits were registered in §352.

Both models use only the 19 native channels shared by PROTEUS and Scherer. Both apply common-average
reference, the same 4–40 Hz postfilter, 12 fixed bands and full/two-half covariance blocks. Model A reads
diagonal log-covariance; model B reads its full weighted upper triangle. Dimensions are 684 and 6,840.
This changes spatial information and capacity together; it cannot isolate physiological spatial structure.
Both use source-only scaling and fixed binary logistic regression, C=.05, with balanced class weights.
Only PROTEUS labels fit models. There is no external fitting, calibration, model selection or teacher.

The reference uses each training session's covariance statistics. Inference uses the delivered prefix
and a PROTEUS-only mean log-covariance prior, k=20, reset each day. All input tasks contribute unlabeled
session statistics; MI is excluded only from binary fitting and metric. Both corpora use the frozen
four-second exported cue+1→+5 windows and µV/120 Hz. Original acquisition filters/hardware differ;
applying identical postfilters cannot undo that historical difference.

Before external accuracy, the collector requires complete ten-person source-held predictions, all eight
pinned source orders, person exclusion and source covariance BA≥60%, with at least seven people above50%.
Only then does it evaluate 1,420 external WORD/SUB rows from nine people/two days/142 recovered runs.
Foreign orders use the independently verified recipe; they have no historical source-order artifact.
They are frozen before accuracy. Intervals resample people and nested days/orders.

The current external crop includes the last0.25seconds of the visual cue. Spanish letter fluency,
subtraction range, clinical population, hardware and absent artifact-rejection masks qualify interpretation.
The three-second preparation period cannot provide a four-second silent pre-cue control. There is no
such assumed control in this study. These external people were already used in historical training;
only the new PROTEUS-only learner provides training-person separation. The campaign is not blind.

## Preserved failures and repair

- 3119536 failed before Python import: login-node `/tmp` was unavailable on the compute node.
  One second/two CPUs; no scientific measurement. Source moved to Lustre.
- 3119551 failed bitwise prefix probabilities after exact source training/order guards. Nine seconds/two CPUs.
- Source-only diagnostic3119554 passed: features were bitwise equal; batched covariance logistic-regression
  probabilities differed by1.11e−16, with no changed labels. Actual one-window sklearn prediction restored
  bitwise equality. The producer now uses that one-window path; no tolerance or fitted mathematics changed.
- v2 smoke3119555 passed actual100sourcewindows, exact training twins, independent seven-window covariance,
  seventeen-window prefix and all eight source-order guards. No accuracy was computed.
  Features0.973s, paired fit0.068s, measured-process RSS approximately0.53GiB; projected full wall276seconds.
  [Full smoke record](pair352_smoke.json) pins the source hash and code.

Full3119558 uses immutable `/mnt/lustre/work/gehler/gfh527/ni26_pair352_20261117_v2`, CPU2/8GiB/40min,
BLAS/OpenMP one thread, zero GPU. Cost/hash/RSS guards precede production; the owning Python feature
module hash is checked before and after. Source-held and external banks precede external gold.
Outputs stay Galvani `runs/diag/external_pair352_v2/`; generated models/data are not committed.
No subsequent matrix or package promotion is authorized by this study.
