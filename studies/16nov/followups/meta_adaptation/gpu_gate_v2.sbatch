#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s307-gpu-gate
#SBATCH --output=logs/s307-gpu-gate-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
for f in source.sha256 pinned.sha256 cpu-src.sha256 dispatch.sha256; do sha256sum -c "$f" >/dev/null; done
cmp source.sha256 ../gpu_smoke_v2/source.sha256
test "$(sacct -X -n -P -j 3115209 -o State,ExitCode | head -1)" = 'COMPLETED|0:0'
sacct -X -n -P -j 3115209 -o JobID,Elapsed,State,ExitCode
"$PY" -B -I studies/16nov/followups/meta_adaptation/verify_gpu_v2.py
set -o noclobber
printf 'GPU pilot 3115209 verified by %s\n' "$SLURM_JOB_ID" > gpu-gate.ok
