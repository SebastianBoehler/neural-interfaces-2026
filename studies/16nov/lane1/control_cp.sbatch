#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-l1-control-cp
#SBATCH --output=logs/control-cp-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
order=${orders[$SLURM_ARRAY_TASK_ID]}
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
for base in B2_pool B2_s345 B2_s012 B2_s0 B2_s1 B2_s2; do
    "$PY" -B -I studies/14nov/newmember/s281_cp.py control "$order" "$base"
done
