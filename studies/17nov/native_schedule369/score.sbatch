#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani,a100-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:18:00
#SBATCH --job-name=ni26-schedule369-dev
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
base=/mnt/lustre/work/gehler/gfh527
repo="$base/neural-interfaces-2026"
reference="$base/ni26_native366_exact_release_v1_20261117"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py="$repo/.venv/bin/python"
sha256sum -c source.sha256
(cd "$reference" && sha256sum --quiet -c package.sha256)
(
  cd "$base/ni26_s318_coral_fbl_20261117"
  "$py" replay/check.py
) > replay.log
grep -Fx OK replay.log
"$py" variants.py "$reference/package" "$PWD"
"$py" fixtures.py capture
apptainer exec --cleanenv --containall --no-home --no-mount hostfs   --env OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1   --bind "$PWD:/work:rw,$reference/package:/stock:ro,$repo/data/proteus_windows.npz:/data.npz:ro"   --pwd /work "$repo/runs/cloud/ni26_worker.sif" python replay_cp.py
"$py" fixtures.py score
sha256sum -c source.sha256
