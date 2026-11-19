#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:40:00
#SBATCH --job-name=ni26-laneE-train
#SBATCH --output=logs/train-%A_%a.out
# Usage (in the Lane E work dir): sbatch --array=0-5 train_extra_seeds.sbatch [smoke]; task 0-2 eye seed 3-5, 3-5 rest seed 3-5
set -euo pipefail
work=${SLURM_SUBMIT_DIR:?}; cd "$work"; mkdir -p members logs
t=${SLURM_ARRAY_TASK_ID:?}; kind=eye; [ $t -ge 3 ] && kind=rest; seed=$((3 + t % 3)); extra=${1:-}
out=members; [ "$extra" = smoke ] && out=members_smoke
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NI26_WS="$work/code"
sha256sum --quiet -c code.sha256
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -I -B "$work/code/lane_e/train_extra_seeds.py" $kind $seed "$work/$out" $extra
