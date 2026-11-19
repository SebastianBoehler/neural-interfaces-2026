#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-c18
#SBATCH --output=c18-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
root=$base/ni26_s318_coral_fbl_20261117
repo=$base/neural-interfaces-2026
parent=$base/ni26_friday_causal_rebuild_20261118/within0_terminal_shared_v1
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
(
    cd "$parent"
    echo "ad7f624831d9e25ab636a7096a07a6e64999bdf47e7dcea3994e4210b84d8326  source.sha256" | sha256sum --check --status
    sha256sum --check --status source.sha256
)
(
    cd "$root"
    sha256sum --check --status "$parent/owner.sha256"
    sha256sum --check --status "$parent/recognition.sha256"
    sha256sum --check --status "$parent/complete_terminal_inputs.sha256"
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$parent:$work"
if [[ ${1:?smoke, complete or score} == score ]]; then
    sha256sum --check --status scoring.sha256
    exec "$repo/.venv/bin/python" -B score.py
fi
exec "$repo/.venv/bin/python" -B worker.py "$1"
