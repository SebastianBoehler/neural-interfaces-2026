# §281 private Galvani orchestration

These scripts prepare the already registered new-person chain and CP outputs. They do not submit by themselves. The workspace is `/mnt/lustre/work/gehler/gfh527/ni26_s281_codex_20261116`; its `smoke` child must contain its own copies of the code and every input, including the v6.6 ZIP. Inputs must be private copies, without symlinks or hardlinks. The shared interpreter is read-only: each process runs `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I`; it is never activated, modified or compiled.

The workspaces need the runner's replay inputs: `data/proteus_windows.npz`, `runs/replay_data/`, all eight §257 references and their SHA manifest, the pinned permutations, `runs/rundetect/decisions.npz` and `proj_d6_r1.npz`, the pinned v6.6 ZIP, and `runs/s281_leo_inputs/remote.sha256`. Copy the 96 corresponding base NPZs into **one** of `runs/ttt/probs18` or `runs/diag/ttt/probs18` per order. Also copy the recording-order `ttt7` base files, every `LPC` base and r15c250 stream, and the eye, rest, core, Hilbert, teacher and GE shuffled inputs expected by `run_s_v66.py`. The runner performs its own STRICT input and archive checks. Do not put other task outputs in these workspaces.

Preflight hashes the pinned package, permutations, archive references and all 96 Leonardo streams. It rejects linked files, duplicate stream locations, pre-existing outputs and missing stage files. It checks that every public all-LOSO session has fewer than 250 windows: the saved study's `t >= 250` cap differs from the package's `t > 250` cap, so parity is not established for longer sessions. It creates only private output/log directories. The full workload must not start until the isolated smoke has finished and its result has been inspected.

## Smoke

The copied driver **must run from the child workspace**, so its `__file__` and current directory resolve to the same isolated root. Stage the complete copies first, then execute on Galvani:

```bash
cd /mnt/lustre/work/gehler/gfh527/ni26_s281_codex_20261116/smoke
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/15nov/s281/galv_v66_job.py preflight-smoke
cd ..
mkdir -p runs/s281_v66_orch/slurm
sbatch -J ni26s281-smoke studies/15nov/s281/galv_v66_task.sbatch smoke
```

Wait for the smoke job to finish with `COMPLETED 0:0`; inspect its Slurm log. It runs control, zero_coral and S1 for within0; then B2_s0 CP for all three. It checks their CP provenance and control/zero full-stage equality, without computing a metric. Its output is confined to the smoke child and is never used by the full wave.

## Full wave

After a successful inspected smoke, preflight the separate full workspace. Submit these commands manually and retain each returned Slurm job ID. The second and third arrays use `afterok` for the entire preceding array; no task runs if the preceding array fails. No automatic retry or existing-output skip is provided.

```bash
cd /mnt/lustre/work/gehler/gfh527/ni26_s281_codex_20261116
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/15nov/s281/galv_v66_job.py preflight
sbatch --parsable -J ni26s281-ctl --array=0-7%8 studies/15nov/s281/galv_v66_task.sbatch control
sbatch --parsable -J ni26s281-arm --array=0-47%8 --dependency=afterok:CONTROL_JOB_ID studies/15nov/s281/galv_v66_task.sbatch treatment
sbatch --parsable -J ni26s281-cp --array=0-55%8 --dependency=afterok:TREATMENT_JOB_ID studies/15nov/s281/galv_v66_task.sbatch cp
```

Replace each uppercase job-ID placeholder with the previous returned ID. Control tasks map to eight orders. Treatment tasks map to `(zero_coral,S1,S2,S3,S4,S5) × eight orders`. CP tasks map to `(control,zero_coral,S1,S2,S3,S4,S5) × eight orders`. Each CP task starts six separate one-thread B2 base processes, waits for all six and reports every exit; their logs are distinct files under `runs/s281_v66_orch/logs/`. A missing output, launch error, nonzero child exit or existing output fails the task. Each wave is limited to eight concurrent CPU allocations. The Slurm account, partition and time limit are prepared from existing Galvani patterns and still need a cluster-side check before submission.

The CP outputs are not a submission or a score. Before scoring, collect and verify all outputs, the per-order routing traces and their input provenance. The scorer separately requires full-stage zero checks for all orders and bases. Real package gate behavior and returning-person routed tails remain separate validation tasks.
