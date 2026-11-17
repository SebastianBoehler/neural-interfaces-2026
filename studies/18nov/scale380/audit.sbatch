#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,v100-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-audit380
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
root=/mnt/lustre/work/gehler/gfh527
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
sha256sum --quiet -c audit.sha256
(cd "$root/ni26_scale380_v2_20261118" && sha256sum --quiet -c source.sha256)
"$root/neural-interfaces-2026/.venv/bin/python" -B check_cpu.py "$PWD/audit379.py" "$root/ni26_scale380_v2_20261118"
sha256sum --quiet -c audit.sha256
