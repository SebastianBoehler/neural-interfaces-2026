#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --job-name=ni26-external356
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_base=/mnt/lustre/work/gehler/gfh527
task_python="$task_base/neural-interfaces-2026/.venv/bin/python"
sha256sum -c source.sha256
if [[ "$1" == full ]]; then
    "$task_python" -c 'import json; x=json.load(open("smoke/before_labels.json")); assert x["scope"]=="smoke P01 only" and x["proteus_labels_opened"] is False and x["zero_identity"] and x["mi_decisions_exact"] and x["cp_score_domains_exact"] and x["conservative_full_projection_seconds"]<1200 and x["max_rss_kib"]<6*1024**2'
    cd "$task_base/ni26_s318_coral_fbl_20261117"
    "$task_python" replay/check.py > "$SLURM_SUBMIT_DIR/replay.log"
    grep -Fx OK "$SLURM_SUBMIT_DIR/replay.log"
    cd "$SLURM_SUBMIT_DIR"
    "$task_python" run.py --out full
elif [[ "$1" == smoke ]]; then
    "$task_python" run.py --out smoke --smoke
else
    exit 2
fi
sha256sum -c source.sha256
