#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:15:00
# Slurm-only runtime qualification of an exact packaged Solver in the worker image.
set -euo pipefail
root=$1
pkg=$2
out=$3
mode=${4:-full}
if [[ "$mode" == smoke ]]; then
  test -d "$pkg"
  bind_artifact="$pkg:/package:ro"
  actor=(python -B -u /source/check.py /package /wave.npz)
elif [[ "$mode" == full ]]; then
  test -f "$pkg"
  bind_artifact="$pkg:/artifact.zip:ro"
  sha256sum "$pkg" > "$out/archive.sha256"
  actor=(bash -e -c 'test ! -e /work/unpacked
    python -m zipfile -e /artifact.zip /work/unpacked
    python -B -u /source/check.py /work/unpacked /wave.npz
    exec python -B -u /source/runtime_check.py --pkg /work/unpacked --data /wave.npz --out /work/runtime.json')
else
  echo "Invalid worker mode $mode" >&2
  exit 2
fi
base=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=2,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,HF_HUB_OFFLINE=1,MPLCONFIGDIR=/work,PYTHONPATH=/official,REVE_POSITIONS_PATH=/reve_positions \
  --bind "$root/cleanslate/teamA:/source:ro,$bind_artifact,$out:/work:rw,$base/2026-competition:/official:ro,$base/data/proteus_windows.npz:/wave.npz:ro,$HOME/mne_data:/reve_positions:ro" \
  --pwd /work "$base/runs/cloud/ni26_worker.sif" \
  "${actor[@]}"
