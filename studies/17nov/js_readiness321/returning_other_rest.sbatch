#!/bin/bash
#SBATCH --job-name=ni26-js321-returning-other-pilot
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
families=(fb fbl fbraw)
index=${SLURM_ARRAY_TASK_ID:?}
family=${families[$((index/11))]}
within=$((index%11))
if ((within<5)); then fold=P04_s23; seed=$((within+1)); else fold=P10_s23; seed=$((within-5)); fi
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python returning_other.py "$family" "$fold" "$seed"
