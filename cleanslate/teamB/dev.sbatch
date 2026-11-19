#!/bin/bash
#SBATCH --account=gehler
#SBATCH --job-name=csB-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%A_%a.out
# One team B dev fold in the worker image (same binds as harness/fold.sbatch). Env: CS_REL, CS_RUN, CS_ORDERS (';' list),
# CS_VARIANTS, CS_ENV_SEMI (NAME=VAL;... for train and stream). Row k of $CS_RUN/jobs.tsv: mode<TAB>held
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
IFS=$'\t' read -r mode held < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CS_RUN/jobs.tsv")
CS_ENV=${CS_ENV_SEMI:-}; CS_ENV=${CS_ENV//;/,}
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=4,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work,PYTHONPATH=/cs/harness:/cs${CS_ENV:+,$CS_ENV} \
  --bind "$CS_REL/cleanslate:/cs:ro,$CS_RUN:/work:rw,$base/neural-interfaces-2026/2026-competition:/official:ro,$base/neural-interfaces-2026/data/proteus_windows.npz:/data/proteus_windows.npz:ro,$base/ni26_csB_20261120/insp:/ext:ro" \
  --pwd /work "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" \
  python -B /cs/teamB/dev.py --mode "$mode" --held "$held" --orders "${CS_ORDERS//;/,}" --run /work --variants "${CS_VARIANTS:-base}"
