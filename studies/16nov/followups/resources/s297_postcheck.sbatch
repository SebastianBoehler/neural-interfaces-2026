#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/postcheck-%j.out
#SBATCH --error=logs/postcheck-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c resource_unmasked.sha256
sha256sum -c resource_postcheck.sha256
: "${EXPECT_CPUS:?}"
test "${SLURM_CPUS_PER_TASK:?}" = "$EXPECT_CPUS"
test -z "${SLURM_JOB_GPUS:-}"
test -z "${SLURM_STEP_GPUS:-}"
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then echo 'CUDA mask unexpectedly set' >&2; exit 1; fi
info=$(scontrol show job "$SLURM_JOB_ID")
printf '%s\n' "$info" | grep -E 'JobId=|Partition=|NumNodes=|ReqTRES=|AllocTRES='
if printf '%s\n' "$info" | grep -Eq 'AllocTRES=[^ ]*gres/gpu'; then echo 'unexpected GPU allocation' >&2; exit 1; fi
srun --exclusive --ntasks=1 --cpus-per-task="$EXPECT_CPUS" --cpu-bind=verbose,cores bash resource_postcheck_step.sh
