#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-c29-source
#SBATCH --output=c29-source-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
owner=$base/ni26_friday_c20e_native_v4_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source-fit.sha256
(cd "$base" && sha256sum --check --status "$owner/parents.sha256")
(cd "$prior" && sha256sum --check --status inputs.sha256 && sha256sum --check --status members.sha256)
grep -Fx OK "$prior/replay-proof.txt"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/tmp,PYTHONDONTWRITEBYTECODE=1 \
  --bind "$work:/work:rw,$prior/cap:/package:ro,$repo/2026-competition:/official:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B source29.py "${1:?}"
