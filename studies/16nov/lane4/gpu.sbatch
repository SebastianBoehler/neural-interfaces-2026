#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane4
#SBATCH --output=runs/lane4/job-%A_%a.out
set -euo pipefail
if [[ ${SMOKE:-0} == 1 ]]; then
    exec bash studies/16nov/lane4/task.sh 0 --smoke
fi
exec bash studies/16nov/lane4/task.sh "${SLURM_ARRAY_TASK_ID:?missing array id}"
