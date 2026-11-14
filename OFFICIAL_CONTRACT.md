# Track 2 PROTEUS linear baseline contract

Checked 2026-10-14 against competition revision
`a66bc23d1a0e686b895c093a642b45c74970758b` and NeuroAI tag `v0.4.0`.
This records the official implementation; it does not claim a completed run.

## Baseline and split

- `tracks/bci_decoding/training.yml` selects `Torch-Linear`,
  `BCI[study=dreyer2026proteus]`, `BCI-decoding[training=True]`, seed 42.
- The model takes the temporal mean of every EEG channel and applies
  `nn.Linear(n_chans, n_classes)`, followed by argmax for prediction.
- Training is 30 epochs with Adam, learning rate 0.01, and unweighted
  `CrossEntropyLoss`. Dataset training batch size defaults to 64.
- This is the official default linear example. `MeanLogReg` is a separate
  supplied linear example; reproducing it would be a different experiment.
- PROTEUS is NEMAR `nm000290` v1.0.0: 112 runs, 14 sessions, 10 participants,
  source EEG-BIDS at 500 Hz, CC BY 4.0. Command runs use Graz or BrainHero.
- NeuralBench partitions subjects 60/20/20 into train/validation/test, with
  `valid_random_state: 33` and `test_random_state: 33`.
- Training and warm-up both set run seed 42. That seed does **not** change
  split random states: `load_task` overrides `Data.seed`, while the task's
  `study.split` retains both explicit random states. `Data.prepare()` calls
  `study.run()` before using `Data.seed` for loader/sampler RNG streams.
- The official adapter supplies only the training loader to `fit`; the test
  loader belongs to the objective. The linear solver does not use validation.

## Preprocessing and streaming inference

- One window starts 1 second after each command cue and lasts 4 seconds.
- Notch filtering: 50 and 60 Hz. Band-pass: 0.1–75 Hz at recorded rate.
  Resample to 120 Hz and multiply volts by 1e6 to obtain microvolts.
  No scaling or clamping is configured.
- Evaluation tensor is float32 `(1, 41, 480)`. Output is one integer label
  `(1,)`: 0 motor imagery, 1 mental subtraction, 2 word generation.
- One stream is one participant/session. The pinned public code delivers runs
  by BIDS run number and windows in recording time, but the sealed backend does
  NOT: since the rescoring of 10 Nov the delivery order is fixed and is not
  recording order (organisers, 13 Nov; verbatim in docs/STATE.md section 2). The
  scope of the shuffle (whole session, per interface, per run) is not stated.
  Graz and BrainHero can share session state.
- The evaluator deep-copies the loaded model at each session boundary and
  calls optional `reset_state()`. No model state crosses session boundaries.
  FAQ: "State, windows, statistics, and parameter updates must not cross the
  session boundary." That includes module-level variables (clocks, caches,
  counters) that change later predictions.
- `predict` receives only EEG, without labels or subject/interface metadata.
  Torch-Linear has no adaptation state.

## Metric and packaging

- Ranking metric: balanced accuracy within participant/session/interface
  cells, then equal average across cells: `balanced_accuracy_cell_mean`.
- Additional diagnostics: session-average balanced accuracy, pooled balanced
  accuracy, plain accuracy, number of classes, and number of cells.
- NeuralBench's `bal_acc_stream_mean` lacks interface separation, so it is
  not numerically interchangeable with the Codabench ranking metric.
- Training exports `tracks/bci_decoding/outputs/Torch-Linear/submission.py`
  and `weights.pt`. The ZIP must contain those files at its root.
- `submission.py` contains `class Solver(CompetSolver)` and the inference
  model. Additional Python files are unsupported. Read artifacts via
  `meta["submission_dir"]`; the submitted directory is read-only.
- Evaluation is inference-only. No training or dataset download occurs there.
  The competition YAML sets a 3,600-second evaluation limit.

## Official commands from the competition checkout

```sh
benchopt install tracks/bci_decoding --config tracks/bci_decoding/training.yml -y
benchopt prepare tracks/bci_decoding --config tracks/bci_decoding/training.yml
benchopt run tracks/bci_decoding --config tracks/bci_decoding/training.yml
```

After training, replay the exported model without selecting training:

```sh
COMPET_SUBMISSION_DIR="$PWD/tracks/bci_decoding/outputs/Torch-Linear" \
  benchopt run tracks/bci_decoding \
  -d "BCI[study=dreyer2026proteus]" \
  -s "$PWD/tracks/bci_decoding/outputs/Torch-Linear/submission.py" \
  --seed 42
```

The optional `tools/run_docker.py --track bci_decoding --submission <zip>
--data <data-root>` replays ingestion and scoring inside the worker image.
It requires Docker and builds the image; a native Benchopt replay is not
proof that this Docker path or the remote A100 worker passed.

## Cost and verification limits

- No published Torch-Linear score, preparation runtime, download size, or
  local training runtime was found in the reviewed official docs. Measure
  those during the run instead of comparing to invented targets.
- Torch-Linear contains 126 parameters for 41 channels and three classes;
  dataset download/filtering/caching can dominate this small model's cost.
- Real data use requires four NeuroAI packages pinned to 0.4.0 and
  `nemar-py>=0.3.1`. The worker requirements also pin `exca==0.5.29`.
- The native adapter selects CUDA when available and CPU otherwise; it does
  not select Apple MPS. macOS CPU validation does not validate CUDA behavior.
- Synthetic checks have no scientific score meaning and are not a PROTEUS
  baseline. A fixed 41-channel checkpoint must replay on matching dimensions.
- Public warm-up test labels are available. Keep train/test subjects disjoint
  and do not optimize the initial replication on warm-up test labels.

## Primary sources

- [Competition files at the checked revision](https://github.com/neural-interfaces26/2026-competition/tree/a66bc23d1a0e686b895c093a642b45c74970758b)
- [Training config](https://github.com/neural-interfaces26/2026-competition/blob/a66bc23d1a0e686b895c093a642b45c74970758b/tracks/bci_decoding/training.yml)
- [Linear solver](https://github.com/neural-interfaces26/2026-competition/blob/a66bc23d1a0e686b895c093a642b45c74970758b/tracks/bci_decoding/solvers/torch_linear.py)
- [Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/a66bc23d1a0e686b895c093a642b45c74970758b/codabench/pages/scientific_faq_bci.md)
- [NeuralBench stream config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml)
- [NeuralBench data and RNG implementation](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/data.py)
- [Submission guide](https://github.com/neural-interfaces26/2026-competition/blob/a66bc23d1a0e686b895c093a642b45c74970758b/codabench/pages/participate.md)
