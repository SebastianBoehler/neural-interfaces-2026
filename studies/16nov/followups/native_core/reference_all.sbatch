#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-s301-refall
#SBATCH --output=logs/reference-all-%j.out
#SBATCH --error=logs/reference-all-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum source.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export NI26_ROOT="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH
sha256sum -c source.sha256 >/dev/null
sha256sum -c reference_all.sha256
printf '%s  %s\n' 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 data/proteus_windows.npz | sha256sum -c -
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/reference_all.py
