#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f31-score-v2
#SBATCH -o score-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
archive=$base/ni26_friday_s1c_order_20261119
origin=$base/ni26_s323_normsplit_20261117
f14=$base/ni26_friday_s1c_training_20261118
f16=$base/ni26_friday_s1c_distill_20261118
f17=$base/ni26_friday_s1c_phase_20261118
PY=$base/neural-interfaces-2026/.venv/bin/python
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c recovery.sha256
(
    cd "$archive"
    sha256sum --quiet -c - <<'PINS'
832bf16a1af007cadef7742a2ee3bb08cd27c51a9dccf4ebd9a691f6be38e24d  actor.sha256
12d8348244e7769d2e80e8e07158952185ad242e3037d63addf4d78cec368d8a  score.sha256
a40d5e40ec256384aad788d6926e56e4183a048c93995918e81c37e09aa05b22  before_labels.sha256
628b6fc0a57200a8c0e7d7061e876a008a6d7d2869494b51bca84184868afe3e  whole_results.json
PINS
    sha256sum --quiet -c actor.sha256
    sha256sum --quiet -c score.sha256
    sha256sum --quiet -c before_labels.sha256
)
(cd "$origin" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
printf '%s\n' '503080a78e0c7b66f51e03451a93fd149613888dfa98328bce7c2795e63e018f  score.sha256' | \
    (cd "$f16" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256 && sha256sum --quiet -c before_labels.sha256)
printf '%s\n' '7ed98f96ba6b05165a1bc035a20af0d7dc516b9b1d61c82533abcc85ecbf2372  actor.sha256' | \
    (cd "$f17" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256)
export PYTHONPATH="$archive:$origin:$origin/experiments:$origin/studies/14nov/newmember:$f14:$f16:$f17"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
exec "$PY" -B collect_order.py
