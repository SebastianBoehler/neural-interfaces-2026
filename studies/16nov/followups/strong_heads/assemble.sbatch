#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s308-assemble
#SBATCH --output=logs/assemble-%j.out
#SBATCH --error=logs/assemble-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source studies/16nov/followups/strong_heads/runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sha256sum -c source.sha256 >/dev/null
test "$(sha256sum source.sha256 | cut -d' ' -f1)" = "$S308_PRODUCER_SHA"
srun --cpu-bind=cores "$PYTHON" -B -I studies/16nov/followups/strong_heads/assemble.py
