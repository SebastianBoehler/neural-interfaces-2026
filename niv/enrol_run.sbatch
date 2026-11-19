#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=02:30:00
#SBATCH --job-name=ni26-niv-ret
#SBATCH --output=logs/niv-ret-%x-%A_%a.out
# Usage: sbatch --array=0-8 enrol_run.sbatch SYS MODE      SYS = enrolled package tag (e.g. cap250), MODE = ret | fa
#  ret: package k, person k, visit 2 only, 16 orders (full and within, 8 each)
#  fa : package k, the 8 other people, both visits, order full:0 (false-recognition / zero-routing check)
set -euo pipefail
sys=${1:?}; mode=${2:?}
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"; sha256sum --quiet -c inputs.sha256
k=${SLURM_ARRAY_TASK_ID:?}; tag=${mode}_${sys}_k$k; mkdir -p out/$tag
if [ "$mode" = ret ]; then persons=$k; visits=2; arms=$(python3 -c "print(','.join(f'{s}:{i}' for s in ('full','within') for i in range(8)))")
else persons=$(python3 -c "print(','.join(str(j) for j in range(9) if j!=$k))"); visits=1,2; arms=full:0; fi
out=out/$tag/p${k}_c0.npz; [ -e "$out" ] && { echo "exists $out"; exit 1; }
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work,NIV_VISITS=$visits \
  --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$work/scherer_rows.tsv.gz:/rows.tsv.gz:ro,$work/zips/enrolled/${sys}_k$k.zip:/pkg.zip:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B niv/worker.py /pkg.zip "/work/$out" "$persons" "$arms"
