#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-ocular-head-fit
#SBATCH --output=eye-cal-head-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
fit=$base/ni26_friday_return_eye_cal_fit_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
(cd "$fit" && sha256sum --quiet -c fit_source.sha256)
(cd "$base" && sha256sum --quiet -c "$fit/fit_inputs.sha256")
export NI26_WS="$repo"
export PYTHONPATH="$repo/experiments:$repo/scripts:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$repo/.venv/bin/python" -B fit_calibration.py
