# Research block: 17 November, early morning

Actual start: 01:48:22 UTC / 02:48:22 CET, 17 November 2026.
Actual end: 02:09:39 UTC / 03:09:39 CET; substantive block elapsed21minutes17seconds.
Scope: reinterpret the completed overnight blocks, audit representation coverage,
check current primary research and released implementations, and prepare distinct next tests.

## What the experiments change

The complete reports in [STATUS_17NOV.md](STATUS_17NOV.md) establish no new package gain.
They do change which comparisons would be informative:

- §313 imaginary spectral features beat matched real features by 1.974 official
  points, but not the converged BASE reliably. Combining all spectral features
  hurts that BASE. This supports selective information tests, not adding dimensions indiscriminately.
- The [baseline audit](strong_baseline_comparability_17nov.md) finds that §313 BASE
  closely matches §305's newly converged head, not §306's historical original.
  Nearly identical source objectives do not make different stopping paths equivalent decoders.
  Implicit regularization is a hypothesis; a feature/cache defect is unsupported.
- §314 temporal derivatives improve their weak whole decoder by 2.182 official
  points. Their guarded §315 addition changes the current system by −.110 points;
  its person/order interval crosses zero. Possible causes include redundant
  information, weak standalone quality, calibration and downstream interaction.
  The result does not identify which cause dominates or close temporal learning.
- §304 conditional CORAL30 swap improves the current control by .0545 official
  points, with an interval crossing zero. Its advantage over unconditional10 mixes
  conditioning and strength. §318 preserves matched singles and interaction controls.
- §305 native fitting and the faster §309 device solve establish different scopes.
  Language, optimizer convergence, runtime and held-person accuracy must remain separate.

The next portfolio therefore changes representation and supervision while retaining
bounded combinations. More jobs on one representation would not establish wider coverage.

## Literature and implementation checks

### Visual pretraining: an existing lead now made executable

[SeeEEG](https://arxiv.org/html/2611.17031), submitted 12 November 2026,
was already reviewed on 16 November. This block does not count rediscovery as new research.
Its frozen vision backbone receives channel-row waveform or scalogram renderings;
task learning adjusts the display and readout. Frozen-encoder benchmarks and random-backbone
controls motivate the prior, but do not establish superiority to fine-tuned EEG models
or our PROTEUS system. Source-person counts and tasks also differ.

New operational check: the official [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
provides a 21M-parameter ViT-S/14, 384-dimensional embeddings, public weights and an
Apache2 model license. It accepts larger patch-aligned images. Shapes that are not
multiples of14 can crop; padding must be explicit. The inspected model-card blob is
`21b9bf295c8cab14e782e1a7a1d051be9e501088`; license blob
`5471dc10377b76db85a2feca7a99a7eef4980ba8`. The official
[backbone loader](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py),
blob `af5fd95d38eb33f73a85b92a94a715773a4e70bc`, loads checkpoints strictly.
The downloaded checkpoint and pinned repository revision still need their own hashes.

Our proposed adaptation uses DINOv2, not the paper's DINOv3; it is not a reproduction.
At patch14, an image with41 rows of patches and480 samples padded to490 has1435
patch tokens;16 rows have560. Attention cost grows with token count, even if the
backbone is frozen. Neither count is a measured runtime or proof of worker feasibility.

**Decision:** assign §319 a no-score real-window preflight on Galvani, followed by
a bounded source-only pilot if it passes. Preserve amplitude with source-fixed
scaling and an explicit mean/power path. Compare pretrained and random versions of
the same encoder, plus a matched simple waveform control. Compare full-channel
rendering with an explicitly declared source-only projection. Measure warm B1 cost,
memory, padding, offline imports, active gradients, reset and prefix invariance.
Register numeric hypotheses and source selection before opening any pilot labels.
An outer matrix requires its own prospective complete-matrix registration.

### Forward-only learned aggregation: a different learning mechanism

[Backpropagation-Free Transformations](https://arxiv.org/html/2601.07556)
was first submitted12January2026 and revised17August2026; it is not a recent-days paper.
The method predicts through a fixed set of transformed inputs or feature masks,
then uses a source-trained reliability ranker to weight the predictions. Its feature
variant shares an encoder. Published motor-imagery and drowsiness results do not
validate our three classes, shuffled delivery or official equal-cell metric.
Temperature changes and ordinary normalization also need matched controls.
The authors link [DeepTransferEEG](https://github.com/sylyoung/DeepTransferEEG);
the assigned worker must inspect the actual implementation before adopting details.

Our inference: learned branch reliability may be useful without adapting network
weights on unreliable target labels. This is not an independent representation.
Earlier confidence selectors, augmentation and softening already exist here;
source-held-person ranker supervision and fixed branch identities are the distinctions to audit.

**Decision:** assign §320 a code/coverage audit, then a bounded source-only pilot if
the mechanism is viable. Use a strong raw encoder, retain unmasked mean/power, and
train the ranker from source-person cross-fitted predictions. Compare unchanged
output, uniform same-mask aggregation, learned aggregation and neutral temperature
controls. Freeze or explicitly control BatchNorm state across views. Require fixed
mask identity, no-future/reset guards and measured B1 cost. Do not open outer labels
or silently attach the experiment to an unfinished §312/317 matrix.

### Offline tabular priors: readiness improved, still a lower-priority lead

The current official [TabPFN repository](https://github.com/PriorLabs/TabPFN)
supports offline3.5 weight paths; the hosted Thinking upgrade and offline model
availability are different questions. This refines the earlier API-only readiness
description. Code and checkpoint licenses differ, and some model access requires
account/terms acceptance. No terms were accepted and no EEG data was sent to a provider.
Existing TabICLv2/TabPFN coverage means this is not a newly discovered family.

A later trial should compare compact physiologically selected features with a
matched linear learner and explicit source-only selection. The known damage from
compressing the strong44k representation is a warning against attributing a weak
baseline win to a tabular prior. Checkpoint rights/access and B1 resource cost remain unverified.

### Recency and incomplete sources

Focused13–16November searches did not surface a newer directly actionable primary
paper in this block. This is a search result, not evidence that none exists.
Earlier reviews already cover November EEG encoders, graph masking, alignment,
metric failure modes and interdisciplinary training mechanisms.
The OpenReview BTTA-DG lead was abstract-only because the full PDF returned a
browser challenge. No detailed claim or implementation decision relies on it.

## Portfolio and next decisions

| Question | State at assignment | Changed assumption | Next decisive evidence |
| --- | --- | --- | --- |
| §312 raw recording pretraining | Separate CPU/GPU matrices underway | Supervision can use previously unused recording intervals | Complete guarded matrices, then pinned full-system consumers |
| §313 spectral structure | Whole result complete; integration pending | Cross-channel phase structure may complement amplitude | Complete integration, then examine redundancy/calibration before selected follow-up |
| §316 common-anchor meta-adaptation | Fresh target matrix/consumer underway | Learn update rules while fixing a verified trained anchor | Complete guarded comparison; never relabel failed independent §307 |
| §317 continued external cognitive supervision | Source pilot passed;60-fold production queued | Auxiliary cognitive learning persists during task training | Complete A/B/C matrix and private full-system consumer |
| §318 conditional CORAL + FBL2 | Smoke passed; capacity-controlled waves queued | Small effects may combine through different mechanisms | Singles, joint effect and interaction under matched controls |
| §319 visual prior | Newly assigned; no score | Non-EEG pretraining plus signal display may supply another representation | Licensed hashed artifact, real-window feasibility, matched source pilot |
| §320 forward-only reliability | Newly assigned; no score | Source-learned view reliability may avoid harmful gradient updates | Coverage/code audit, controlled source pilot, B1 cost |

Assignments are execution states, not completed experiments. New pilots initially
use at most one allocation each and obey the shared32-running/272-active ceiling.
All heavy work remains on Galvani. Existing source snapshots and jobs are preserved.
Fresh workers still read canonical project history; these are not blind or cross-provider replications.

Read each complete guarded result independently. Queue justified questions without
waiting for unrelated matrices. Keep source selection, whole-model metrics,
current-system effects and sealed ranking separate. No upload follows automatically.
