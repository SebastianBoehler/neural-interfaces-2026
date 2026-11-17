#!/bin/bash
#SBATCH --job-name=ni26-native353-neural-source
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
running=$(squeue --me -h -r -t RUNNING | wc -l)
active=$(squeue --me -h -r | wc -l)
if ((running>32 || active>244)); then exit 1; fi
base=/mnt/lustre/work/gehler/gfh527
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
 --bind "$base:$base:ro" --bind "$PWD:/out:rw" --pwd /out \
 "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" python neural_fixture.py \
 --helpers "$base/ni26_js321_20261117_returning_pair_cpu1" \
 --package "$base/ni26_js321_20261117_active5/draft_base" \
 --data "$base/neural-interfaces-2026/data/proteus_windows.npz" --out /out/reference
sha256sum -c source.sha256 >/dev/null
