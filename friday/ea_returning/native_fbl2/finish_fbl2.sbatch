#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-c21r-seal
#SBATCH --output=c21r-finish-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
replay=$base/ni26_friday_return_fbl2_native_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
source=${2:?native source digest}
cd "$work"
sha256sum --quiet -c source.sha256
(cd "$replay" && sha256sum --quiet -c source.sha256)
test "$(sha256sum "$replay/source.sha256" | cut -d ' ' -f1)" = "$source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "${1:?seal or score}" in
    seal) exec "$repo/.venv/bin/python" -B verify_fbl2.py seal "$source" ;;
    score) exec "$repo/.venv/bin/python" -B score_fbl2.py "$source" ;;
    *) exit 2 ;;
esac
