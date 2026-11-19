#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-c20r-collect
#SBATCH --output=c20r-finish-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
replay=$base/ni26_friday_return_c20_native_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
(cd "$replay" && sha256sum --quiet -c source.sha256)
test "$(sha256sum "$replay/source.sha256" | cut -d ' ' -f1)" = 7b9fc805bc65a9b9f73df20b807f49fb28167fbb25cdcfa31bfad4452f7eaa4e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "${1:?collect or score}" in
    collect) exec "$repo/.venv/bin/python" -B verify_return.py collect ;;
    score) exec "$repo/.venv/bin/python" -B score_return.py ;;
    *) exit 2 ;;
esac
