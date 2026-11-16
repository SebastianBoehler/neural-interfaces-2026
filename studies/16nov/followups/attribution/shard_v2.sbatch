#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-300-order
#SBATCH --output=logs/300-order-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s300_shards_20261116/system
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c followup-score.sha256
sha256sum -c attribution-src.sha256
sha256sum -c shard-src.sha256
sha256sum -c shard-v2-src.sha256
sha256sum -c attribution-registration.sha256
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
if [[ ${1:-} == smoke ]]; then
 "$PY" -B -I studies/16nov/followups/attribution/shard.py within0 --smoke
 "$PY" -B -I studies/16nov/followups/attribution/check_shard.py
else
 "$PY" -B -I studies/16nov/followups/attribution/shard.py "${orders[${SLURM_ARRAY_TASK_ID:?}]}"
fi
