#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=ni26-s301-ref
#SBATCH --output=logs/reference-%j.out
#SBATCH --error=logs/reference-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" NI26_ROOT="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
unset PYTHONPATH
sha256sum -c source.sha256 >/dev/null
sha256sum -c reference.sha256
printf '%s  %s\n' 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 data/proteus_windows.npz | sha256sum -c -
printf '%s  %s\n' 6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f runs/shuffle_perms.npz | sha256sum -c -
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/native_core/reference.py
