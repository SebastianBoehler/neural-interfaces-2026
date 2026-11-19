#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-c16-score
#SBATCH --output=c16-score-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
inner=$base/ni26_friday_returning_native_inner_v2_20261119
prior=$base/ni26_js321_20261117_returning_pair_cpu1
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status fit.sha256
sha256sum --check --status replay.sha256
sha256sum --check --status scoring.sha256
(cd "$inner" && sha256sum --check --status "$work/parents.sha256" && sha256sum --check --status source.sha256 && sha256sum --check --status inputs.sha256)
(cd "$prior" && sha256sum --check --status source.sha256)
(cd "$base" && sha256sum --check --status "$work/score_inputs.sha256")
grep -Fx OK "$base/ni26_aux431_20261118/runs/aux431/replay.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$inner:$prior:$repo/experiments:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
exec "$repo/.venv/bin/python" -B collect.py
