# §352 Frozen PROTEUS-only WORD/SUB external stress test

Status: **complete**, Galvani 3119558 passed. Actual collection verified 17 November 2026 at 13:13:27 UTC. This is a binary external condition-discrimination
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
pinned source orders, person exclusion and source covariance BA≥60%, with at least seven people above 50%.
Only then does it evaluate 1,420 external WORD/SUB rows from nine people/two days/142 recovered runs.
Foreign orders use the independently verified recipe; they have no historical source-order artifact.
They are frozen before accuracy. Intervals resample people and nested days/orders.

The current external crop includes the last 0.25 seconds of the visual cue. Spanish letter fluency,
subtraction range, clinical population, hardware and absent artifact-rejection masks qualify interpretation.
The three-second preparation period cannot provide a four-second silent pre-cue control. There is no
such assumed control in this study. These external people were already used in historical training;
only the new PROTEUS-only learner provides training-person separation. The campaign is not blind.

## Preserved failures and repair

- 3119536 failed before Python import: login-node `/tmp` was unavailable on the compute node.
  One second/two CPUs; no scientific measurement. Source moved to Lustre.
- 3119551 failed bitwise prefix probabilities after exact source training/order guards. Nine seconds/two CPUs.
- Source-only diagnostic 3119554 passed: features were bitwise equal; batched covariance logistic-regression
  probabilities differed by 1.11e−16, with no changed labels. Actual one-window sklearn prediction restored
  bitwise equality. The producer now uses that one-window path; no tolerance or fitted mathematics changed.
- v2 smoke 3119555 passed 100 actual source windows, exact training twins, independent seven-window covariance,
  seventeen-window prefix and all eight source-order guards. No accuracy was computed.
  Features 0.973 s, paired fit 0.068 s, measured-process RSS approximately 0.53 GiB; projected full wall 276 seconds.
  [Full smoke record](pair352_smoke.json) pins the source hash and code.

Full 3119558 uses immutable `/mnt/lustre/work/gehler/gfh527/ni26_pair352_20261117_v2`, CPU2/8GiB/40min,
BLAS/OpenMP one thread, zero GPU. Cost/hash/RSS guards precede production; the owning Python feature
module hash is checked before and after. Source-held and external banks precede external gold.
Outputs stay Galvani `runs/diag/external_pair352_v2/`; generated models/data are not committed.
No subsequent matrix or package promotion is authorized by this study.


## Result and interpretation

All 10 source-held models, all 16 arm/order banks, exact prefix/reset/interleaving guards, source
exclusions and model immutability passed before the producer wrote `before_labels.json`.
The collector then scored the source banks and checked the registered sanity condition. It passed:
source binary BA power **62.7733%**, covariance **66.8994%**, covariance above chance for **10/10 people**.
P04 was 55.1892/59.7875%; P10 59.9501/67.8501% (power/covariance). These are new-person source-held
controls, not returning-person validation. Only after that check did the collector open external labels.

| External binary decoder | Person/day/order mean BA | 80% people-nested interval |
| --- | ---: | ---: |
| Diagonal log-covariance (power) | 53.2602% | 51.9581–54.6007% |
| Full log-covariance | 59.0303% | 56.7250–61.3569% |
| Full minus diagonal | +5.7701 percentage points | +3.9583 to +7.5409 |

Full covariance beats diagonal for eight of nine people. Individual full-covariance means range
50.9375–65.1562%. The means weight nine people equally and their two days equally, over all eight
orders. Intervals resample people and nested days/orders. They do not correct prior campaign selection.
[Complete result](pair352_result.json), [source sanity](pair352_source_sanity.json) and
[before-label proof](pair352_before_labels.json) retain all people/day/order values and identities.
All executed Python source hashes match the committed source.

This establishes a usable **external binary condition-discrimination instrument** for this source-frozen
19-channel learner. Some WORD/SUB discrimination transfers across the distinct clinical people, and the
full-covariance readout transfers better than its diagonal control. It does not establish universal
cognitive rules, a stronger current PROTEUS decoder, an ensemble gain or a rank forecast. Covariance
changes capacity as well as spatial information; cue, clinical and acquisition limits remain.
The moderate 66.9% common-channel source control also limits comparison with the stronger 41-channel
current system. There was no source-only 41-channel binary comparator or physiological negative control.

Actual producer time 640.98 seconds/process CPU 638.48 seconds; complete allocation 646 seconds on two CPUs,
MaxRSS 2,847,008 KiB (about 2.72 GiB). All four preliminary jobs plus the full allocation used 1,332 allocated
CPU-core-seconds, **0.370 CPUh/0 GPUh**. The smoke's 276-second projection underestimated the complete
10-fold/eight-order causal-reference and one-window prediction workload; no performance claim follows.
All owned jobs are complete. No additional fit, feature grid, package change or upload was launched.
