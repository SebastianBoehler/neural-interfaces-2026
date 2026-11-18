# One raw-waveform decoder

The campaign target is the highest eligible sealed Track 2 rank. Whole-model balanced accuracy on people absent from training measures generalization in this exploration lane. Select the complete submission using both unseen and returning-person validation, independent external checks and measured worker time. The model receives the official preprocessed float32 EEG, one window of 41 channels and 480 samples at 120 Hz. F7 learns the sensor and time filters, temporal layers, class head and session readout jointly. The deployment path is a single small model. The auxiliary head and moving-average target exist during training only.

## Evidence that changes the design

- The first six Fourier-based architectures and perturbation conditions reached 38.82–43.92% as whole decoders. Every insertion trailed stock. P10 lost 1.39–2.40 percentage points under all twelve model/seed pairs. P08 also lost under all twelve, although its median standalone waveform score (54.16%) was the best of the ten people. That contrast favors measuring whole-decoder accuracy and complementarity separately. [Held results](wave/results-18nov.md)
- The earlier source-held, causal single-model filter-bank baseline reached 59.05% on the same ten public people and eight orders. That selected historical result is a useful architectural reference. The complete historical cap250 system scored 76.10%. Both anchors matter. [Single-model normalization results](../../neural-interfaces-2026/evidence/next_wave_16nov/norm323.md)
- F5 recovered some information about which wrong class stock predicted. Its genuine corrections performed worse than exact cap250 on independent Scherer people. This argues for first making the waveform representation useful on its own. [Exact external validation](errorfit/external-result.md)

## Current tests

F7 compares supervised training with the same model learning to predict the continuous latent representation of a masked second of EEG. The clean control executes the same auxiliary prediction but sends its gradient only to the auxiliary head. A genuine improvement on unseen people would justify expanding the source corpus. Broader masked EEG pretraining and continuous latent prediction are credible techniques. The relevant question is transfer to these three commands with this inference budget. [REVE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html) · [Continuous latent EEG prediction](https://arxiv.org/abs/2608.11656)

F8 uses controlled synthetic EEG during source training. It retains a real trial's task and timeline, then perturbs how its sensors co-vary using another source session. Its control matches sensor variance while omitting the cross-channel change. Generate and consume perturbations inside each training fold. Report gains only on real held EEG. A 2026 class-conditioned generative study reported small, classifier-dependent gains, so a full generator needs the same comparison to a strong simpler transform. [EEG generation study](https://arxiv.org/abs/2607.22733)

## A label-free version of fast adaptation

The transferable idea from meta-learning is to learn the update on training people. Give the model shuffled source episodes and learn how the first few unlabeled windows should change its next prediction. F9 tests a learned session normalization rate against the identical network with fixed-rate and blank updates. Report the first delivered window, the next nine, and later contributions to the official score. This identifies cold-start performance separately from evidence gained from a prefix. A new person's evaluations supply no reward or task labels.

For a richer generator, condition on the task instruction and the acquisition protocol, then fit on licensed, paired recordings. Three command names alone add the same class identity already available during training. Paired descriptions can add meaning when merging studies with different instructions. Text can specify the conditions and an agent can implement and verify the generator. The measured waveforms and real held outcomes provide the calibration and test. [EEG and language instruction tuning](https://arxiv.org/abs/2409.00101)

The [queue](QUEUE.md) contains training and scorer dependencies. The [plot source](figures/plot.py) includes a new model only when both seed result files are available against the same authenticated stock.
