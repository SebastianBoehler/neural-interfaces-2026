#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH --time=3
#SBATCH -J ni26-f33-a100-canary
#SBATCH -o a100-%j.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
r=$b/neural-interfaces-2026
origin=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f17=$b/ni26_friday_s1c_phase_20261118
f33=$b/ni26_friday_s1c_covariance_v2_20261119
data=$b/ni26_friday_wave_band_20261118
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c canary.sha256
(cd "$origin" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
printf '%s\n' '984e6a7c7776fefca6fd70924ddc135b92dd53ab3fdffd206fc435ccaa5df70f  actor.sha256' | \
    (cd "$f33" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
printf '%s\n' '7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372  actor.sha256' | \
    (cd "$f17" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
export PYTHONPATH="$origin:$origin/experiments:$f14:$f16:$f17:$f33:$data"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$r/.venv/bin/python" -B canary.py
