#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-s304
#SBATCH --output=logs/304-%A_%a.out
#SBATCH --error=logs/304-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
sha256sum -c coral304.sha256 > /dev/null
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [[ ${1:-} == smoke ]]; then
  "$PY" -B -I studies/16nov/followups/coral_repair/smoke.py
else
  id=${SLURM_ARRAY_TASK_ID:?}
  arms=(plain unconditional10 conditional10 conditional30)
  "$PY" -B -I studies/16nov/followups/coral_repair/produce.py "${arms[$((id / 60))]}" "$(((id % 60) / 10))" "$((id % 10))"
fi
