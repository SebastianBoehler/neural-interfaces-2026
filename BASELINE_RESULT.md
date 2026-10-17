# Official PROTEUS linear baseline: local validation complete

Completed on 14 October 2026 (Europe/Berlin). Nothing was submitted to Codabench.

## Implementation reproduced

- Official repository revision: `a66bc23d1a0e686b895c093a642b45c74970758b`.
- Official configuration: `tracks/bci_decoding/training.yml`, unchanged.
- Solver: `Torch-Linear`, unchanged; temporal channel means, linear 41-to-3 layer.
- Training: 30 epochs, Adam learning rate 0.01, cross-entropy, run seed 42.
- Public dataset: PROTEUS, NEMAR `nm000290`, version 1.0.0.
- Preprocessing, channels, windows and participant split come from NeuroAI 0.4.0.
- Native macOS ARM CPU execution; exact packages are in
  `runs/baseline/requirements.lock.txt`.

## Data audit

| Split | Participants | Windows | Class 0 / 1 / 2 |
|---|---|---:|---|
| Train | P01, P02, P03, P06, P07, P10 | 1,521 | 510 / 507 / 504 |
| Validation | P05, P08 | 379 | 126 / 127 / 126 |
| Test | P04, P09 | 755 | 249 / 252 / 254 |

The participant sets are disjoint. All sampled windows in all three loaders
were checked for finite float32 signals, 41 channels, 480 samples, and integer
class labels in 0–2. Test batches contain one window. Sampling frequency is
120 Hz; signals retain the official microvolt units.

## Measured public-test result

| Metric | Training evaluation | Packaged inference replay |
|---|---:|---:|
| **Balanced accuracy averaged over participant/session/interface cells** | **37.7543528303%** | **37.7543528303%** |
| Balanced accuracy averaged over sessions | 37.7110156908% | 37.7110156908% |
| Balanced accuracy pooled over windows | 37.7230456635% | 37.7230456635% |
| Plain accuracy | 37.7483443709% | 37.7483443709% |

There are 8 scoring cells. The cell-average metric is the competition ranking
metric; the others are diagnostics. This is one run, not a multi-seed estimate
or a hidden-test result. No official published Torch-Linear numeric target was
found, so this reproduces the official implementation rather than claiming
agreement with a published score.

Benchopt reports 7.97 seconds for the training/evaluation run. The separate
inference-only ingestion replay took 4.89 seconds. These times exclude initial
environment installation, download and preprocessing.

## Package and validation

`runs/baseline/proteus-torch-linear.zip` contains exactly `submission.py` and
`weights.pt` at its root (2,651 bytes).

SHA-256: `548b00d469c2c01607b7114c794b3162e398655c5d6d8014c5eb34e205af663e`.

Passed checks:

- Solver source matches the official source byte-for-byte.
- All 126 model parameters are finite; weight/bias shapes match three classes
  and 41 input channels.
- ZIP integrity and root layout are valid.
- Official ingestion reloads the ZIP's extracted weights in inference-only mode.
- All four metrics match the training evaluation within absolute tolerance 1e-12.
- Official scoring writes finite `scores.json` values using the warm-up phase map.
- Extracted submission files are read-only and remain byte-identical after replay.
- The official checkout has no tracked modifications.

Evidence: `runs/baseline/data_audit.json`, `validation.json`, `train.log`,
`ingestion.log`, `scoring/scores.json`, and the raw result Parquet files.

## Limitations and resolved setup issue

This is native CPU replay, not the Linux/A100 worker container or official
Codabench scoring. No submission quota was consumed.

The optional `NEURALBENCH_EXTRACT_CPUS` setting is incompatible with this
task's `LabelEncoder`: the upstream helper adds an unsupported `target.infra`
field. The local environment unsets it and uses the unchanged official defaults.
The failed attempt is retained in
`runs/baseline/prepare-extraction-setting-failure.log`.

Benchopt logs a `git describe` warning because the upstream repository has no
tags. Training and replay still exit successfully; the exact commit SHA is
recorded independently. MNE also emits deprecation notices for the official
montage name. Neither required changes to the baseline or its preprocessing.

Rerun with `bash scripts/run_baseline.sh`. Source references and the contract
are documented in `OFFICIAL_CONTRACT.md`.
