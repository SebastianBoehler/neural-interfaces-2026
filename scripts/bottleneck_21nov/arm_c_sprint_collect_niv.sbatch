#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
work=/mnt/lustre/work/gehler/gfh527
repo="$work/neural-interfaces-2026"
export APPTAINERENV_PYTHONPATH="$PWD:$PWD/src/scripts/last_push:$PWD/src/scripts/bottleneck_21nov"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec apptainer exec --bind "$work" "$repo/runs/cloud/ni26_worker.sif" python -B \
    scripts/bottleneck_21nov/arm_c_sprint_scherer.py "$repo/data/scherer2015/windows.npz" \
    "$work/ni26_ba_six_fc_20261121/niv-rows.tsv.gz" "$work/ni26_quarter_final_20261121/niv-results" \
    out/tape-score out/scherer.json
