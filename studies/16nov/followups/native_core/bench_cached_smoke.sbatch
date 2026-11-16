#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s301-cachesmoke
#SBATCH --output=logs/cached-smoke-%j.out
#SBATCH --error=logs/cached-smoke-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source studies/16nov/followups/native_core/bench_cached_env.sh
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/bench_cached_smoke.py
