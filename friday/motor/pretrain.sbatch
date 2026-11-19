#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 5
#SBATCH -J ni26-motor-pre
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
py=$b/.venv/bin/python
source_root=$b/data/physionetmi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c --status pretrain.sha256
kinds=(motor_transfer motor_null)
seeds=(460 1460)
if [[ $1 == smoke ]]; then
    for kind in "${kinds[@]}"; do
        "$py" pretrain.py "$kind" 460 "$source_root" --smoke
    done
else
    task=${SLURM_ARRAY_TASK_ID:?}
    "$py" pretrain.py "${kinds[task % 2]}" "${seeds[task / 2]}" "$source_root"
fi
