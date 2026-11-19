#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-c31c-seal
#SBATCH --output=seal-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
order=$base/ni26_s318_coral_fbl_20261117/runs/shuffle_perms.npz
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c source.sha256
sha256sum --quiet -c seal.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$repo/.venv/bin/python" -B seal_tape.py "$repo/data/proteus_windows.npz" "$order" "$base/ni26_friday_c20s_native_v1_20261119"
