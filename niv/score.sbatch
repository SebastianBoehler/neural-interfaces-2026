#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=02:30:00
#SBATCH --job-name=ni26-niv
#SBATCH --output=logs/niv-%x-%A_%a.out
# Usage (in the release dir): sbatch --array=0-35 score.sbatch ZIP_NAME TAG [smoke]
# task = person*4 + chunk; chunk c scores order seeds 2c,2c+1 of both schemes (full, within).
set -euo pipefail
zipname=${1:?}; tag=${2:?}; mode=${3:-run}
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
work=${SLURM_SUBMIT_DIR:?}
cd "$work"; mkdir -p out/$tag logs
sha256sum --quiet -c inputs.sha256
t=${SLURM_ARRAY_TASK_ID:-0}; person=$((t/4)); chunk=$((t%4))
if [ "$mode" = smoke ]; then arms="full:0"; person=5; out=out/$tag/smoke.npz
else arms="full:$((2*chunk)),full:$((2*chunk+1)),within:$((2*chunk)),within:$((2*chunk+1))"; out=out/$tag/p${person}_c${chunk}.npz; fi
[ -e "$out" ] && { echo "exists $out"; exit 1; }
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work \
  --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$work/scherer_rows.tsv.gz:/rows.tsv.gz:ro,$work/zips/$zipname:/pkg.zip:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B niv/worker.py /pkg.zip "/work/$out" "$person" "$arms"
