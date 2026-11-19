#!/bin/bash
#SBATCH --account=gehler
#SBATCH --job-name=cs-fold
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%A_%a.out
# One LOPO/returning fold: train.py on its roster, then stream the held sessions through the package.
# Called by run_lopo.py; env: CS_REL (release dir with cleanslate/), CS_TEAM, CS_RUN (run dir), CS_ORDERS, CS_ENV (extra
# NAME=VALUE,... passed to train and stream, e.g. ablation switches). Row k of $CS_RUN/jobs.tsv = this task:
# ('-' = empty first_only: tab is whitespace for read) mode<TAB>held<TAB>train_people<TAB>first_only<TAB>sessions
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
t=${SLURM_ARRAY_TASK_ID:?}
CS_ENV=${CS_ENV_SEMI:-}; CS_ENV=${CS_ENV//;/,}; CS_ORDERS=${CS_ORDERS//;/,}
IFS=$'\t' read -r mode held people first sessions < <(sed -n "$((t+1))p" "$CS_RUN/jobs.tsv")
[ "$first" = "-" ] && first=""
out=$CS_RUN/${mode}_${held}; mkdir -p "$out"
sha256sum "$base/neural-interfaces-2026/data/proteus_windows.npz" | cut -c1-16 > "$out/data.sha16"
run() { apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=4,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work,PYTHONPATH=/cs/harness:/cs:/official/tracks/bci_decoding${CS_ENV:+,$CS_ENV} \
  --bind "$CS_REL/cleanslate:/cs:ro,$CS_RUN:/work:rw,$base/neural-interfaces-2026/2026-competition:/official:ro,$base/neural-interfaces-2026/data/proteus_windows.npz:/data/proteus_windows.npz:ro" \
  --pwd /work "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" "$@"; }
if [ ! -e "$out/pkg/.trained" ]; then
  run python -B /cs/$CS_TEAM/train.py --people "$people" ${first:+--first-session-only "$first"} --out /work/${mode}_${held}/pkg
  touch "$out/pkg/.trained"
fi
[ -e "$out/pkg/submission.py" ] || { echo "train.py produced no submission.py"; exit 3; }
run python -B /cs/harness/stream.py --pkg /work/${mode}_${held}/pkg --held "$held" --sessions "$sessions" \
  --orders "$CS_ORDERS" --out /work/pred_${mode}_${held}.npz
