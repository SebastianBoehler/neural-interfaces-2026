#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/p2-compare-%j.out
#SBATCH --error=logs/p2-compare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c resource_p2.sha256
export CUDA_VISIBLE_DEVICES="" NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
test -z "${SLURM_JOB_GPUS:-}"
info=$(scontrol show job "$SLURM_JOB_ID")
printf '%s\n' "$info" | grep -E 'JobId=|Partition=|NumNodes=|ReqTRES=|AllocTRES='
if printf '%s\n' "$info" | grep -Eq 'AllocTRES=[^ ]*gres/gpu'; then echo 'unexpected GPU allocation' >&2; exit 1; fi
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I compare_resource.py
