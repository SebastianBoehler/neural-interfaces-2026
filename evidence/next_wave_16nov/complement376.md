# §376 · Is learning from another decoder's errors useful?

The nested log-bandpower proxy found some of the windows where the complete cap250 system fails. On ten held people, both models erred on 16.398% of the official cell/class-balanced measure. Independent placement of the same number of proxy mistakes within each class and cell predicts 14.977%. That difference of **+1.422 percentage points** [80% people/order interval +0.974,+1.910] measures error localization. A fixed training objective must still turn that information into a better complete system. [Fidelity result](complement376b_result.json)

The proxy alone scored 43.273% on held people. Selecting equally many low-confidence windows in each true class and cell captured 14.949% joint errors. Actual proxy errors improved localization by +1.450 pp [+0.767,+2.147]. The independence contrast was positive for all ten people. Positive confidence contrasts covered eight people, with P02 and P10 below zero. Matching used held labels for this retrospective check. [Prediction freeze and lineage](complement376b_before_labels.json)

## Three exclusion boundaries

1. For each outer test person, fit the proxy nine times. Every inner fit uses eight people and predicts the ninth. The outer person's labels, rows and normalization statistics never enter those fits.
2. Train a residual member on the nine outer-source people. Preserve both original §323 frozen population FBCNet bodies, true source scaling and seed-specific checkpoints. The source labels can affect the new head, matrix weights and source feature normalization. The outer person enters only after training. All 90 nested proxy probability banks replicated bitwise in §376b.
3. Apply both residual heads to the held person, average the two seed probabilities, and pass that tensor through the existing full-system stages, shuffled orders and six dependent CP bases. Score the unmodified RF, untouched control, causal gates and zero-dose endpoints alongside treatment. P04 and P10 here are new-person tests. The independent conditional returning assessment remains separate.

## What each objective changes

| Arm | Source use |
|---|---|
| RF | Original §323 P0F seed-averaged member. Every held emission must exactly reproduce §339 RF. |
| O | Learn a bounded three-class logit correction with ordinary class/cell-balanced cross-entropy. |
| E | Triple weights on nested proxy mistakes. Preserve total training mass in each cell and class. |
| C | Assign the same weight values by proxy uncertainty within each cell and class. |
| S | Shuffle the same values within each cell and class. |
| A | Minimize the loss of a 75% proxy + 25% member arithmetic probability mixture. |
| G | Minimize the loss of the corresponding logarithmic opinion pool. |

For proxy probabilities $p$, member probabilities $q$, and true class $y$, A's loss is $-\log(.75p_y+.25q_y)$. Its gradient on the member's logits is the ordinary loss gradient multiplied by $.25q_y/(.75p_y+.25q_y)$. Interchanging the two incorrect proxy classes leaves this gradient unchanged. G can use their relative probabilities. All 20 source-fold/seed checks verified A's gradient identity within 1.12×10⁻¹⁶. Swapping the two wrong classes changed G's gradient by at least 0.0136. The two heads use the same features, initial zero correction, optimizer, per-cell weights and update count. [Source producer](../../studies/17nov/complement376/mixture/mixture.py)

## Immutable runs

| Work | Galvani workspace under /mnt/lustre/work/gehler/gfh527 | Slurm |
|---|---|---|
| §376 producer | ni26_complement376_v2_20261117 | P01 3123026, other people 3123027 |
| §376b fidelity | ni26_complement376b_v1_20261117 | 3123051 |
| §376c mixture heads | ni26_complement376c_v1_20261118 | P01 3123054, other people 3123055 |
| §376c full-system consumer | ni26_consumer376c_v1_20261118 | Setup 3123089, stage 3123090, parent verification 3123091, initial full cells 3123092, gate verification 3123128, remaining cells 3123093, scorer 3123094 |

All study runs used CPU. Earlier failed §376 provenance lookup and full-system numerical guards remain in the ledger. The official submission, submodule and frozen A100 parity artifacts are separate controls. [Active checkpoint](OVERNIGHT_18NOV.md) · [Preregistration and failures](../../docs/LEDGER.md)

## Full system on ten held people

All 56 full cells, six personalization bases per cell, eight orders, stock,
zero, RF endpoint, row, route, metric and historical replay checks passed.
The original C/S within0 jobs failed exact gate-input equality on cn212/cn203.
Their intermediates are preserved. On cn057 those two calculations passed
unchanged after moving earlier scratch outputs aside. Array 3123194 then
completed exactly the 42 remaining cells. Scorer 3123195 completed in 52 s
on 18 November. It consumed 336 final personalization files. This comparison
uses new-person routing under public development sessions.

| Model or paired effect, percentage points | Deployment B2_pool | Six-base average |
|---|---:|---:|
| Untouched control | 76.2793 | 76.2177 |
| Ordinary O | 76.3817 | 76.3543 |
| Error E | 76.4174 | 76.3441 |
| Confidence C | 76.3532 | 76.3569 |
| Within-cell/class shuffled S | 76.3666 | 76.4079 |
| E−O, paired 80% interval | +0.0357 [−0.0513,+0.1371] | −0.0102 [−0.0653,+0.0547] |
| E−C | +0.0642 [−0.0467,+0.1928] | −0.0127 [−0.0654,+0.0515] |
| E−S | +0.0508 [−0.0606,+0.1711] | −0.0637 [−0.1279,+0.0091] |
| E−untouched | +0.1380 [−0.0680,+0.3716] | +0.1265 [−0.0271,+0.2873] |

B2_pool E−O was positive for 6/10 people, with P04 −0.0111 and P10
−0.0415 pp. The standalone E−O gain was +0.0810 [−0.4631,+0.7997].
Evidence remains undecided for error weighting against the matched controls.
The six-base and B2_pool estimates use the same people. Matching produces
a training-side proxy comparison; the proxy never used outer target labels.
[All comparisons](complement376_result.json)

Original result SHA-256 `0e76856ce80421ee64a69f2ce40291fc6ca3975bb52f30b19e588a6b0659ab1f`.

## Complete full-system development result

Training the arithmetic mixture A matched ordinary training O on both the deployment base and the six-base diagnostic. Training the log pool G performed worse than A on the six-base average. The deployment-base comparison between G and A remains uncertain. Both objectives missed their preregistered improvement over O.

| Outcome | B2_pool deployment base | Six dependent bases |
|---|---:|---:|
| Untouched cap250 | 76.2793% | 76.2177% |
| Ordinary O | 76.3817% | 76.3543% |
| Arithmetic A | 76.3788% | 76.3548% |
| Log pool G | 76.3192% | 76.2925% |
| A − O, pp [80%] | −0.0028 [−0.1524,+0.1503] | +0.0004 [−0.0710,+0.0753] |
| G − O, pp [80%] | −0.0625 [−0.1905,+0.0458] | −0.0619 [−0.1482,+0.0148] |
| G − A, pp [80%] | −0.0597 [−0.2472,+0.0953] | −0.0623 [−0.1222,−0.0171] |

On B2_pool the A−O change for P04 was −0.091 pp, and P10 changed −0.097 pp. G−O was +0.182 pp on P04 and −0.059 pp on P10. Both people were excluded entirely from their own producer. Individual member accuracy was 55.859% for O, 55.519% for A and 55.365% for G. The six-base people-equal G−O change was −0.095 pp [−0.175,−0.016]. No new package qualifies from this comparison. [All arms and intervals](complement376c_result.json)

All 25 newly computed full cells passed. Their total allocation was 13.565 CPU core-hours, above the 7.5–12.5 estimate and below the 17.5 cap. The original §376 validator rechecked 336 CP outputs. The separate consumer matched an untouched control on within0 and the full-stage zero on all eight orders. Frozen package and route guards passed before scoring. The scorer completed in 46 seconds. Result SHA-256 is `3ac3653a2125e80116d86f79e041b39a8e54495fff9cae4e2465145ea9a76bbf`. [Scorer](../../studies/17nov/complement376/consumer/score.py)
