#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-c31-score
#SBATCH --output=c31-score-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
parent=$base/ni26_friday_c25q_exact_v2_20261119
prior=$base/ni26_cross411b_20261118
native=$base/ni26_friday_c21e_native_v2_20261119
owner=$base/ni26_friday_c20e_native_v4_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
sha256sum --quiet -c scoring.sha256
(cd "$parent" && sha256sum --quiet -c source.sha256)
grep -Fx OK "$prior/replay-proof.txt"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/parent:/native:/v4:/prior:/experiments:/official \
  --bind "$work:/work:rw,$parent:/parent:ro,$native:/native:ro,$owner:/v4:ro,$prior:/prior:ro,$repo/experiments:/experiments:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B score_probe.py
