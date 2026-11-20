#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=02:30:00
#SBATCH --job-name=ni26-D3-niv
#SBATCH --output=logs/D3niv-%x-%A_%a.out
# Usage (release dir): sbatch --array=0-8 niv_run.sbatch SYS MODE   SYS = hb | D0z | D3f | D3g (enrolled zips/enrolled/SYS_k$k.zip), MODE = ret | fa  (as niv/enrol_run.sbatch)
#  fa : package k holds person k's visit 1; the 8 other people, both visits, order full:0 = wrong-enrolment / false-recognition harm check
#  ret: person k visit 2, 16 orders (B-ret, by-product)
set -euo pipefail
sys=${1:?}; mode=${2:?}
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"
k=${SLURM_ARRAY_TASK_ID:?}; tag=${mode}_${sys}_k$k; mkdir -p out/$tag
if [ "$mode" = ret ]; then persons=$k; visits=2; arms=$(python3 -c "print(','.join(f'{s}:{i}' for s in ('full','within') for i in range(8)))")
else persons=$(python3 -c "print(','.join(str(j) for j in range(9) if j!=$k))"); visits=1,2; arms=full:0; fi
out=out/$tag/p${k}_c0.npz; [ -e "$out" ] && { echo "exists $out"; exit 1; }
export APPTAINERENV_NIV_VISITS=$visits
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work \
  --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$work/scherer_rows.tsv.gz:/rows.tsv.gz:ro,$work/zips/enrolled/${sys}_k$k.zip:/pkg.zip:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B niv/worker.py /pkg.zip "/work/$out" "$persons" "$arms"
