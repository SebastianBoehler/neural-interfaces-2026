#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 3
#SBATCH -J ni26-f19-weights
#SBATCH -o weights-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
origin=$base/ni26_s323_normsplit_20261117
f14=$base/ni26_friday_s1c_training_20261118
f16=$base/ni26_friday_s1c_distill_20261118
f17=$base/ni26_friday_s1c_phase_20261118
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c actor.sha256
sha256sum --quiet -c score.sha256
sha256sum --quiet -c weightproof.sha256
(cd "$origin" && sha256sum --quiet -c source323.sha256)
(cd "$f17" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c before_labels.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$origin:$origin/experiments:$f14:$f16:$f17"
"$base/neural-interfaces-2026/.venv/bin/python" -B weightproof.py
