# Experiment procedure

## Current checkpoint

The unchanged official Torch-Linear baseline is locally validated on PROTEUS.
One seed gives 37.7543528303% cell-averaged balanced accuracy.
Training evaluation and packaged replay agree within 1e-12 on all four metrics.
No published numeric Torch-Linear target was found. This reproduces the source,
not a published score. Native CPU checks do not establish remote worker success.

The snapshot under `evidence/baseline/` records the completed baseline run.
The pinned dependency snapshot lives under `requirements/`.
Reruns write to ignored `runs/baseline/`; archive important runs separately
before rerunning because logs and summary files use the same names.

## Next steps

1. Validate the baseline ZIP in the official Linux worker environment.
2. Review the package and explicitly authorize any Codabench upload.
3. Compare official EEGNet on the same participant split and ranking metric.
4. Evaluate repeated seeds before drawing performance conclusions.
5. Consider REVE or EEGDM only after checking licenses, pretraining data and cost.

EEGDM is a candidate, not a validated improvement. Its epilepsy benchmarks do
not establish transfer to cross-session mental-command decoding.
Paper: [EEGDM](https://arxiv.org/abs/2508.14086).
Code: [jhpuah/EEGDM](https://github.com/jhpuah/EEGDM).

## Required record for each experiment

- Source revision, dependency versions, model and configuration.
- Participant split, training seed and preprocessing.
- All four metrics, with cell-average balanced accuracy as the ranking metric.
- Training time, inference time, hardware and measured peak memory.
- External data and model provenance, licenses and compute estimate.
- Packaged replay result, archive checksum and remaining verification limits.

Do not tune the initial replication against public test labels. Keep the
official submodule pinned. Place our future model changes outside the submodule.
Use the shared baseline split and metric when comparing candidates.
