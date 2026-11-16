#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=ni26-s308-fit
#SBATCH --output=logs/fit-%A_%a.out
#SBATCH --error=logs/fit-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source studies/16nov/followups/strong_heads/runtime.sh
sha256sum -c source.sha256 >/dev/null
test "$(sha256sum source.sha256 | cut -d' ' -f1)" = "$S308_PRODUCER_SHA"
srun --cpu-bind=cores "$PYTHON" -B -I studies/16nov/followups/strong_heads/produce.py "${SLURM_ARRAY_TASK_ID:?}"
