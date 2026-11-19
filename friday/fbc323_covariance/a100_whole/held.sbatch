#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f33a-held
#SBATCH -o held-%A_%a.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
o=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f17=$b/ni26_friday_s1c_phase_20261118
f33=$b/ni26_friday_s1c_covariance_v2_20261119
PY=$b/neural-interfaces-2026/.venv/bin/python
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c actor.sha256
sha256sum --quiet -c score.sha256
(cd "$o" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
(cd "$f17" && sha256sum --quiet -c actor.sha256)
printf '%s\n' '984e6a7c7776fefca6fd70924ddc135b92dd53ab3fdffd206fc435ccaa5df70f  actor.sha256' \
               'd7f3be1774cfc61fa6744755bd53108f0ce62c76f578e65af21aae12479b653e  score.sha256' | \
    (cd "$f33" && sha256sum --quiet -c - && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
export PYTHONPATH="$o:$o/experiments:$o/studies/14nov/newmember:$f14:$f16:$f17:$f33"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
case "${1:?expected smoke, matrix or collect}" in
    smoke) "$PY" -B held_a100.py P01 0 --smoke ;;
    matrix)
        i=${SLURM_ARRAY_TASK_ID:?}
        printf -v held 'P%02d' "$((i%10+1))"
        "$PY" -B held_a100.py "$held" "$((i/10))"
        ;;
    collect) "$PY" -B score_a100.py ;;
    *) exit 2 ;;
esac
