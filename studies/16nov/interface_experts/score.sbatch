#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-306-score
#SBATCH --output=logs/score-%j.out
#SBATCH --error=logs/score-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export EXPERT_ROOT="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${EXPERT_MANIFEST_SHA:?}" "${EXPERT_JOB_ID:?}"
test "$(sha256sum expert.sha256 | cut -d' ' -f1)" = "$EXPERT_MANIFEST_SHA"
sha256sum -c expert.sha256 >/dev/null
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I score.py
sha256sum -c expert.sha256 >/dev/null
