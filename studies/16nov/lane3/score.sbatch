#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-l3-score
#SBATCH --output=logs/score-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c collector.sha256
export LANE3_MODEL_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane3/attempt1
export LANE3_WEIGHTS=/home/gehler/gfh527/.cache/huggingface/hub/models--braindecode--cbramod-pretrained/snapshots/584cdc415913739a05d84bf0c1cb3db397764507/model.safetensors
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/lane3/score.py score
