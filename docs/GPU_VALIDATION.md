# First ML Cloud validation

## Baselines

The pinned official Track 2 source includes EEGNet (20 epochs, Adam, learning
rate 0.001) and REVE (frozen pretrained encoder plus a ridge-classifier head).
Torch-Linear is also CUDA-compatible. REVE is not full encoder fine-tuning.
See the [official solver directory](https://github.com/neural-interfaces26/2026-competition/tree/a66bc23d1a0e686b895c093a642b45c74970758b/tracks/bci_decoding/solvers).

## Initial jobs

Use Tübingen ML Cloud, Galvani, account `gehler`, user `gfh527`.
Remote workspace: `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026`.
Existing jobs belonging to other projects must remain untouched.

1. CPU preparation: four CPUs, 24 GB RAM, maximum 90 minutes. Install isolated
   Python 3.12, CUDA-enabled PyTorch and official worker dependencies; prepare
   PROTEUS and audit the real participant split and tensors.
2. Dependent GPU validation: one A100, four CPUs, 32 GB RAM, maximum 60 minutes.
   Require working CUDA and run a forward/backward kernel check. Train official
   Torch-Linear and EEGNet, seed 42, then package and replay each on GPU and CPU.

The GPU job starts only after successful preparation. No automatic requeue.
Neither job uploads to Codabench. Both use existing university cluster access;
they do not provision a commercial cloud instance.

## Success checks and evidence

- Correct source revision, participant split and real-data shapes.
- Finite model parameters and correct ZIP root layout.
- GPU training scores match GPU packaged inference within 1e-12.
- Read-only extracted submission files remain byte-identical.
- CPU replay succeeds; report metric differences relative to GPU replay.
- Record hardware, driver, CUDA, installed packages, wall time and metrics.
- Record public replay time against the 60-minute limit. This does not predict
  sealed-test runtime because the private dataset size is not established.

Logs and generated files live under `runs/cloud/`, with GPU results in the
Slurm job ID directory. The driver smoke fails explicitly if CUDA is unavailable.
No silent CPU substitution is accepted for GPU training.

This is native Linux validation using CUDA 12.6 wheels. It is not the official
Docker image, whose current Dockerfile defaults to CUDA 13.0. A separate worker
container replay remains necessary before claiming environment equivalence.

## Follow-up gates

After these jobs pass, run EEGNet seeds 42, 43 and 44 with separate immutable
outputs; record training time, inference time and measured GPU memory. Investigate
the observed linear cross-run variation before claiming exact reproducibility.
Then test the official REVE probe, staging its model and electrode-position cache
before network-isolated replay and checking external model/data licenses.
EEGDM follows only after these reference comparisons; it is not an official
competition baseline.

## Submission commands

Run from the remote workspace after synchronizing this repository:

```bash
mkdir -p runs/cloud
prep=$(sbatch --parsable --output='runs/cloud/prepare-%j.out' scripts/cloud/prepare.sbatch)
sbatch --dependency="afterok:$prep" --kill-on-invalid-dep=yes \
  --output='runs/cloud/validate-%j.out' scripts/cloud/validate.sbatch
```

Do not run dependency installation or dataset preparation on the login node.
