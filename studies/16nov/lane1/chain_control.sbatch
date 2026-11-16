#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-l1-control
#SBATCH --output=logs/control-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain
sha256sum -c chain-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
order=${orders[$SLURM_ARRAY_TASK_ID]}
export NM_NAME=control ADD=none SWAP=none ZERO=0 NM_SEEDS=0,1,2,3,4,5
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/14nov/newmember/run_s_v66.py "$order"
