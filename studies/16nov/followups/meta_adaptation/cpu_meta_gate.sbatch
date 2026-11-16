#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s307-cpu-meta-gate
#SBATCH --output=logs/s307-cpu-meta-gate-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
for f in source.sha256 pinned.sha256 cpu-src.sha256 dispatch.sha256 meta-cpu-src.sha256; do sha256sum -c "$f" >/dev/null; done
cmp source.sha256 ../cpu_meta_smoke/source.sha256
cmp meta-cpu-src.sha256 ../cpu_meta_smoke/meta-cpu-src.sha256
test "$(sacct -X -n -P -j 3115210 -o State,ExitCode | head -1)" = 'COMPLETED|0:0'
sacct -X -n -P -j 3115210 -o JobID,Elapsed,State,ExitCode
"$PY" -B -I studies/16nov/followups/meta_adaptation/verify_cpu_meta.py
set -o noclobber
printf 'CPU meta pilot 3115210 verified by %s\n' "$SLURM_JOB_ID" > cpu-meta-gate.ok
