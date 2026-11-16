#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/coral304-continue-%j.out
#SBATCH --error=logs/coral304-continue-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
control=/mnt/lustre/work/gehler/gfh527/ni26_s304_coral_control_v2_20261116
: "${CONTROL_SHA:?}"
test "$(sha256sum "$control/control.sha256" | cut -d' ' -f1)" = "$CONTROL_SHA"
(cd "$control" && sha256sum -c control.sha256 >/dev/null)
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I "$control/continue.py" "$@"
