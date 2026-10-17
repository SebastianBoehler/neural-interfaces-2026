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
