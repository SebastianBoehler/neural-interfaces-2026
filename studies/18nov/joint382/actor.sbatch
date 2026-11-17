#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-galvani,2080-preemptable-galvani,a100-galvani,a100-preemptable-galvani,v100-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-joint382-end
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_DEVICE=cpu OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [[ "$1" == score ]]; then
    (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > replay.log
    grep -Fx OK replay.log
    "$py" -B score382.py "$2" "$3" "$4"
else
    people=(P01 P02 P03 P09)
    "$py" -B actor382.py "${people[${SLURM_ARRAY_TASK_ID:?}]}" "$2" "$3" "$4"
fi
sha256sum -c source.sha256
