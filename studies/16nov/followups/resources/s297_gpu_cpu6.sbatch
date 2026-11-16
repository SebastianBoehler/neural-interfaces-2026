#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0,4%2
#SBATCH --output=logs/p2-%A_%a.out
#SBATCH --error=logs/p2-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c resource_p2.sha256
export CUDA_VISIBLE_DEVICES=""
test "${SLURM_CPUS_PER_TASK:?}" = 6
test -z "${SLURM_JOB_GPUS:-}"
test -z "${SLURM_STEP_GPUS:-}"
info=$(scontrol show job "$SLURM_JOB_ID")
printf '%s\n' "$info" | grep -E 'JobId=|Partition=|NumNodes=|ReqTRES=|AllocTRES='
if printf '%s\n' "$info" | grep -Eq 'AllocTRES=[^ ]*gres/gpu'; then echo 'unexpected GPU allocation' >&2; exit 1; fi
lscpu | grep -m1 'Model name:'
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I -c "import torch; assert not torch.cuda.is_available(); print('CUDA unavailable as required')"
exec bash studies/16nov/followups/geometry/chain_full.sbatch zero
