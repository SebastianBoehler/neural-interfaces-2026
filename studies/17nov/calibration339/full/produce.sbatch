#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s339-fullsource
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_calibration339_full_v2_20261117
sha256sum -c source.sha256 >/dev/null
export FULL_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_calibration339_full_v2_20261117
export NI26_DEVICE=cuda OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="${FULL_ROOT}/runtime-cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
if [[ ${1:-} == largest ]]; then person=P06; seed=0; else
 read -r person seed < <(/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -c 'import json,os;print(*json.load(open("remaining.json"))[int(os.environ["SLURM_ARRAY_TASK_ID"])])')
fi
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B produce_full339.py "$person" "$seed"
