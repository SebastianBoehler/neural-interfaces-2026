# Adaptation papers: primary-source protocol check, 16 November

## MOPC

[Zhang et al., arXiv 2610.30422](https://arxiv.org/html/2610.30422), v1
21 October 2026, reports an IEEE Transactions on Instrumentation and Measurement
publication dated 22 September 2026. The paper is primarily binary motor-imagery
EEG, with preliminary four-class validation. It uses LOSO source models,
five-fold ensembles, per-session alignment, a FIFO of 24 trials, random temporal
masking, feature-neighbor and class-calibrated pseudo-label losses, and one
round-robin batch-normalization affine update after each trial. The paper reports
roughly 40–60 ms per trial on its CPU setup; that is not our worker timing.

Its prediction-before-update schedule transfers to a strictly causal EEG stream,
but the paper does not establish our 41-channel mixed motor/cognitive task,
our shuffled protocol or session-reset packaging. Any reproduction must reset
alignment, memory, optimizer and weights at each competition session boundary.
The method adapts batch normalization, not a 44,239-feature linear head, and its
reported numbers do not validate our proposed gradient meta-training. Source:
[paper methods and experimental configuration](https://arxiv.org/html/2610.30422),
[authors' repository](https://github.com/KunKun-Zhang/MOPC).

## RG-TTA

[Kumar et al., arXiv 2603.27814](https://arxiv.org/html/2603.27814), v1
29 March 2026, studies forecasting with similarity-modulated learning rates,
loss-driven early stopping and selectively reused regime checkpoints. The
reported streaming protocol trains on an initial 720 rows and updates on
successive batches of 750 rows before forecasting the next horizon. Its loss
and checkpoint selection are forecasting-feedback mechanisms; they do not
provide EEG task labels at prediction time. The 10-batch multi-month setting
also does not establish competition-session reset or shuffled-window behavior.
Similarity-based step scaling is a design hypothesis only: a valid EEG form must
use strictly prior, within-session, label-free statistics and an independent
source-trained controller. Reported forecasting MSE is not BCI accuracy.

The controller is a useful conceptual contrast, not the proposed experiment:
§307 would explicitly learn how three unlabeled gradient steps affect later
source-person query labels. Its outer labels are never target-session labels.
