#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-chain376-end
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD/chain" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c finish_v2.sha256
sha256sum -c setup.sha256
(cd chain && sha256sum -c ../runtime.sha256 >/dev/null)
grep -Fx OK replay.log
cd chain
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B ../finish_v2.py "${1:?}"
