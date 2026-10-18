# Scheduled ML Cloud runs

Submitted 17 October 2026, Europe/Berlin. These are Slurm batch jobs, not
completed benchmark results. Queue delays are not included in wall-time limits.

| Job | Purpose | Allocation | Status at submission |
| --- | --- | --- | --- |
| 2974934 | Environment, PROTEUS preparation and data audit | CPU, 4 cores, 24 GB, 90 min maximum | Pending: Priority |
| 2974935 | CUDA check, Torch-Linear and EEGNet training/replay | 1 A100, 4 cores, 32 GB, 60 min maximum | Pending: Dependency |

GPU job 2974935 requires successful completion of job 2974934. An invalid
dependency cancels the GPU job. Neither job automatically requeues.

Project source: `1459dcdcab8019d974a64aeb03bb27132293e660`.
Official submodule: `a66bc23d1a0e686b895c093a642b45c74970758b`.
Host alias: `mlcloud-galvani`. Account: `gehler`. User: `gfh527`.
Workspace: `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026`.

```bash
ssh mlcloud-galvani 'squeue -j 2974934,2974935'
ssh mlcloud-galvani 'sacct -j 2974934,2974935 --format=JobID,State,ExitCode,Elapsed,MaxRSS'
```

Preparation log: `runs/cloud/prepare-2974934.out`.
GPU log: `runs/cloud/validate-2974935.out`.
Successful GPU evidence: `runs/cloud/2974935/summary.json`, per-model
`validation.json`, archives and CPU/GPU replay logs. Absence of a success
summary must not be presented as completed validation; inspect Slurm and logs.

Read [GPU_VALIDATION.md](GPU_VALIDATION.md) for checks and follow-up gates.

## 18 October 2026 outcome and follow-up jobs

Job 2974934 failed after 95 s: `requirements/cloud.txt` replaced the cu126
torchvision with a PyPI build, and importing torchmetrics raised
`operator torchvision::nms does not exist`. Dependent jobs 2974935 and
2975095 were cancelled by Slurm. `prepare.sbatch` now resolves torch,
torchvision and torchaudio together from the cu126 index
(`unsafe-best-match`) and checks the imports before data preparation.

| Job | Purpose | Result |
| --- | --- | --- |
| 2975102 | Fixed preparation and audit | Completed, 4 min 54 s |
| 2975103 | Torch-Linear and EEGNet GPU training/replay | Completed; see below |
| 2975104_[0-2] | Deep sweep, `dev_loso` and `dev_session`, seeds 0–2 | Completed, about 4 min each |
| 2975118_[0-4] | Deep sweep, `all_loso`, seeds 0–4 | See OPTIMIZATION.md |

GPU validation (A100-PCIE-40GB, driver 580.173.02, seed 42, official
split): Torch-Linear 38.12% and EEGNet 35.73% cell mean. GPU and CPU
replays were identical on all four metrics, and submission files stayed
unchanged. Native Linux only; not the official Docker image.

`scripts/cloud/sweep.sbatch` runs `experiments/run_cv.py` as an array (index =
seed) and writes `runs/experiments/results-<job>-<seed>.jsonl`. It reuses
`data/proteus_windows.npz` (SHA-256 `2be4187f…3e1327`, exported locally from
the official pipeline and copied over) or exports it after preparation.
