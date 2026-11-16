#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s311-blas-smoke
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source studies/16nov/followups/head_kernel/blas_runtime.sh
sha256sum -c source.sha256 >/dev/null
test "$(sha256sum source.sha256 | cut -d' ' -f1)" = "$S311_BLAS_SOURCE_SHA"
sha256sum -c runs/build/library.sha256 >/dev/null
grep -m1 'model name' /proc/cpuinfo
srun --cpu-bind=cores "$PYTHON" -B -I studies/16nov/followups/head_kernel/blas_smoke.py
