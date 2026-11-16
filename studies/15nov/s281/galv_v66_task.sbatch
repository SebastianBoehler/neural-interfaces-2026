#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=runs/s281_v66_orch/slurm/%x-%A_%a.out
set -euo pipefail
[[ $# == 1 ]] || { echo 'usage: sbatch galv_v66_task.sbatch {smoke,control,treatment,cp}' >&2; exit 2; }
root=/mnt/lustre/work/gehler/gfh527/ni26_s281_codex_20261116
python=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [[ $1 == smoke ]]; then
    cd "$root/smoke"
    exec "$python" -B -I studies/15nov/s281/galv_v66_job.py smoke
fi
[[ $1 == control || $1 == treatment || $1 == cp ]] || { echo 'invalid phase' >&2; exit 2; }
[[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ ]] || { echo 'missing Slurm task index' >&2; exit 2; }
cd "$root"
exec "$python" -B -I studies/15nov/s281/galv_v66_job.py "$1" "$SLURM_ARRAY_TASK_ID"
