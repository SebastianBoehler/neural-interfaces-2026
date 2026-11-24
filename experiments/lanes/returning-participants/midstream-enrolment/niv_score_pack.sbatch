#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=02:30:00
#SBATCH --job-name=ni26-R4-niv
#SBATCH --output=logs/R4niv-%x-%A_%a.out
# Lane R4: niv/score.sbatch with the four order chunks of one person in sequence (Galvani per-user job limit 300).
# Usage (NI-V release dir): sbatch --array=0-8 niv_score_pack.sbatch ZIP_NAME TAG    task = person; out files p{person}_c{0..3}.npz as the original
set -euo pipefail
zipname=${1:?}; tag=${2:?}
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"; mkdir -p out/$tag logs
sha256sum --quiet -c inputs.sha256
person=${SLURM_ARRAY_TASK_ID:?}
for chunk in 0 1 2 3; do
  arms="full:$((2*chunk)),full:$((2*chunk+1)),within:$((2*chunk)),within:$((2*chunk+1))"; out=out/$tag/p${person}_c${chunk}.npz
  [ -e "$out" ] && continue
  apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work \
    --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$work/scherer_rows.tsv.gz:/rows.tsv.gz:ro,$work/zips/$zipname:/pkg.zip:ro" \
    --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B niv/worker.py /pkg.zip "/work/$out" "$person" "$arms"
done
