#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-D3-nivset
#SBATCH --output=logs/D3nivset-%j.out
# release dir (ni26_laneD3_20261120): hardcbB enrolled with d3 moments per Scherer person (zips/enrolled/hb_k*.zip) and its D3f/D3g variants (zips/enrolled/D3f_k*.zip, D3g_k*.zip)
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"; mkdir -p zips/enrolled
[ -e zips/hardcbB.zip ] || ln -s $base/ni26_niv_v1_20261119/zips/hardcbB.zip zips/hardcbB.zip
[ -e scherer_rows.tsv.gz ] || ln -s $base/ni26_niv_v1_20261119/scherer_rows.tsv.gz scherer_rows.tsv.gz
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=4,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work,NIV_D3=1 \
  --bind "$work:/work:rw,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$work/scherer_rows.tsv.gz:/rows.tsv.gz:ro,$base:$base:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B niv/build_enrolled.py zips/hardcbB.zip zips/enrolled hb
for k in 0 1 2 3 4 5 6 7 8; do for a in D0z D3f D3g; do
  [ -e zips/enrolled/${a}_k$k.zip ] || python3 -B laneD3/build_drift.py $a zips/enrolled/${a}_k$k.zip zips/enrolled/hb_k$k.zip
done; done
