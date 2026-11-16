# Research block: 16 November, late evening

Actual start: 21:51:07 UTC / 22:51:07 CET, 16 November.
Actual end: 2026-11-16 22:06:18 UTC (UTC); elapsed at least 15 minutes.
Scope: reinterpret today's completed experiments, check recent primary publications,
and choose distinct mechanisms rather than repeating one incumbent's weight sweep.
This supplements the earlier 30-minute review; it does not replace its data audit.

## Experimental facts that changed the questions

- §302 temperature2 recovers +.110 points against E05, but remains −.064 against
  unchanged control. The neutral ENU slot is +.0047 against control; ET2−ENU is
  −.0684 [−.1815,+.0731]. Less harmful EEG confidence is not yet useful new EEG evidence.
- §300 places much of the EEG loss before the count/CP tail. It does not identify
  which of the mix, balancing, SessionEM or TSEM causes that earlier loss.
- §301/305 training features and scaler closely match. The original failed head
  parity came with different optimizer stopping quality. Lower training loss is
  a new scientific contrast; it is not proof of better held-person predictions.
- The strong native feature port preserves 44,239 features. The earlier compact
  221-feature alternatives did not isolate language, architecture or adaptation.
  Their negative results cannot close stronger forms of those families.

Evidence: `neutral_302.md`, `repair_302_complete_results.txt`, `attribution_300.md`
when present, and `native_core_s301.md`; read their exact guards and limits.

## Primary-source review and transfer limits

### Adaptation stability

[NeuroAdapt-Bench](https://arxiv.org/abs/2604.16926), submitted18April2026,
revised25August2026, tests Tent, SHOT and T3A on several EEG foundation models.
Gradient adaptation often loses; prototype adaptation is comparatively stable.
Larger adaptation batches do not consistently repair the losses. Its authors do
not establish the causal reason. Its batch/clinical tasks differ from our reset,
single-window competition. A secondary November crawl date is not a November
paper revision. This supports testing update mechanisms with controls, not
closing gradient learning or adopting prototypes as an assumed winner.

[MOPC](https://arxiv.org/abs/2610.30422),21October2026, and
[RG-TTA](https://arxiv.org/abs/2603.27814),29March2026, receive a separate
[method/protocol audit](research_4h_16nov_adaptation.md). The former concerns
binary motor imagery; the latter is forecasting. Neither directly validates
our three-class, unlabelled, shuffled-session adaptation. Forecast feedback and
cross-session memory cannot silently enter an EEG deployment.

### Learning from past predictions

[Model Annealing](https://arxiv.org/html/2611.17524),12November2026, combines
learning-rate reheats, softened memory predictions and disagreement perturbations.
Its image experiments include schedule, label-smoothing and snapshot-distillation
controls. Reheating harms its SGD ResNet; benefits depend on architecture and
optimizer. A transfer experiment should compare equal schedules and training
budgets before crediting memory. Pixel perturbation radii have no EEG meaning.
Our calibration finding motivates this question, but supplies no EEG gain estimate.
Train-only memory would add no necessary inference ensemble.

### Alignment is not a sufficient success measure

[Alignment De-Identifies Without Aligning](https://arxiv.org/html/2611.15204),
11November2026, proves an invariance for affine-invariant distances between
condition means under invertible congruences. Removing subject identity does
not establish improved relational task structure. Its empirical CKA regularizer
does not establish a universal impossibility. Our log-Euclidean features,
dimension-changing projections and nonlinear heads need their own comparisons;
the theorem is not a closure of these methods. Keep task decoding, per-class
reliability and person variation beside any domain-confusion statistic.

[What Balanced Accuracy Cannot Identify](https://arxiv.org/abs/2611.17925),
13November2026, shows why class disappearance can hide inside an aggregate and
why stream/state construction can reverse adaptation effects. We retain the
official metric unchanged. Per-person class recalls, prediction counts and
stage-level margin diagnostics explain losses alongside that metric. Only its
abstract was available through the browser; the PDF fetch exceeded the tool limit.

### New encoders and spatial structure

[NOMAD-EEG](https://arxiv.org/html/2610.35246),26October2026, combines
state-space time processing and coordinate-based channel attention. The released
base is12M parameters; frozen extraction returns10,240 dimensions at200Hz.
Its motor-imagery experiment includes4-second trials, but not our three tasks.
Larger pretraining/model size does not uniformly help in its reported ablation.
A new encoder is a distinct representation lane, not an assumed improvement.
Use the original120Hz model-facing tensor and an explicit current-window-only
120→200Hz internal transform; upsampling cannot restore missing frequencies.

The [repository](https://github.com/AbdulkaderHelwan/NOMAD-EEG) has an Apache2
license. Inspected `features.py` blob b6093e631b25e03b3f1a8661a7fbac3840a13808
uses eval mode, GPU bfloat16 and channel coordinates. `hub.py` blob
f20fb7d2073b2b2f94751e474209385a9c57cbb7 leaves the automatic weights URL as
TODO and points to a OneDrive folder. Code availability is verified; actual
checkpoint access, checkpoint terms, hash, runtime and import compatibility
remain unverified. Do not report a trained NOMAD lane yet.

[CABrain](https://arxiv.org/html/2611.11782),9November2026, replaces arbitrary
channel-stack convolutional position encoding with correlation-graph encoding,
and makes masking harder where redundant channels otherwise reveal targets.
The mechanism transfers from graph learning and masked reconstruction. Its
paper uses200Hz,19-channel pretraining and temporal/spectral paths. No usable
checkpoint/code link was found in the inspected paper. Implementing its idea
from scratch differs from reproducing its pretrained result. Author affiliation
does not prove a leaderboard team uses this model.

Its two components also have an implementation interaction: heavily masked
nodes can erase graph-position information, so the authors trained them in
stages. Their mask library is precomputed. A from-scratch transfer should keep
graph-only, masking-only and staged-combination arms distinct, with equal budgets.

[Dual-branch graph/convolution](https://arxiv.org/abs/2611.16609),12November2026,
tests anatomically grouped graph features with a convolutional branch on
movement-versus-rest. It motivates a spatial mechanism, not a transferable
three-class score. We must retain cognitive-task coverage rather than using
motor cortex alone. Full matched graph-versus-nongraph controls are required.

## Concrete decisions, not family closures

1. §306 tests a source-only inferred-interface gate with jointly trained experts.
   The equally balanced shared head separates expert value from reweighting.
   It revisits a different question than old oracle downstream balancing.
2. §307 trains through the actual unlabelled update on source-person episodes.
   Signed scalar/block update policies can reverse harmful directions; matched
   fixed-update/frozen controls identify whether meta-training adds anything.
3. §308 tests structured nonlinear and random/kernel heads on strong features.
   Identical linear bottlenecks and the full linear head avoid a weak-control win.
   Its current integration is explicitly posterior-only, not a whole new chain.
4. For the following representation wave, verify NOMAD checkpoint access and
   B1 cost, then register a frozen-versus-finetuned new-encoder comparison with
   source-person selection and current-system integration. Graph/masking source
   training is another form if unavailable weights make direct reproduction fail.
5. For a training-mechanism wave, separate equal-budget soft-target memory from
   reheating and from generic label smoothing. Select EEG-valid perturbations on
   source people only. Do not copy image-space radii or assume every view preserves labels.

The source audit still finds no demonstrated interpolation artifact in the target
export. CAR's null direction and trace-derived regularization need explicit power
controls. A fresh source search rediscovered Scherer/BNCI2015-004 and CMx7-MM;
these already exist in our data inventory and must not be sold as new people.
See [the earlier data audit](../research_block_16nov/data_and_transfer.md).

## Retrospective and next review

Several apparent breakthroughs so far were changes against weak controls, lower
training loss, or recovery from a harmful arm. None alone establishes a current
submission gain. The search should therefore vary representation, objective and
inference mechanism independently, then combine complementary validated effects.
Earlier architectures, splits and stage boundaries are revisable. The official
contract and honest source/test separation remain the comparison's foundations.

Read complete arms as they land. Record negative mechanisms, not only signs.
At the next four-hour block, inspect these new outcomes before choosing papers
and tests. Do not fill the research interval with an unchanged bibliography.
