#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f36c-pupils
#SBATCH -o pupils-%A_%a.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
o=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f17=$b/ni26_friday_s1c_phase_20261118
f33=$b/ni26_friday_s1c_covariance_v2_20261119
f33a=$b/ni26_friday_s1c_covariance_a100_whole_20261119
f36=$b/ni26_friday_s1c_crossfade_a100_20261119
c20=$b/ni26_friday_causal_rebuild_20261118/fc20_pupils_v2
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c pupil.sha256
(cd "$o" && sha256sum --quiet -c source323.sha256)
(cd "$f14" && sha256sum --quiet -c score.sha256)
(cd "$f16" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
(cd "$f17" && sha256sum --quiet -c actor.sha256 && sha256sum --quiet -c score.sha256)
verify() (
    cd "$1"
    printf '%s  %s\n' "$2" "$3" | sha256sum --quiet -c -
    sha256sum --quiet -c "$3"
)
verify "$f33" 984e6a7c7776fefca6fd70924ddc135b92dd53ab3fdffd206fc435ccaa5df70f actor.sha256
verify "$f33a" 94598f4eb485b20ea40c0d7c13f4497ed07d04e9e9c062c72953e97460b782bc actor.sha256
verify "$f36" baf7735eb9ca7f225b99ecbf58e6a16a8fea3a13813ae7c2e8434d0fff901510 actor.sha256
verify "$f36" 2734eeee0e398a53ffc5900b303255a2ef193df31d2adfb448b59eb1aba67367 score.sha256
verify "$c20" 5925163a6331fc062a51f80f8262c05f7c12f069cfdba86d9e2cddf1f57d349c source.sha256
export PYTHONPATH="$f36:$o:$o/experiments:$o/studies/14nov/newmember:$f14:$f16:$f17:$f33:$f33a:$c20"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=$b/neural-interfaces-2026/.venv/bin/python
case "${1:?expected smoke or matrix}" in
    smoke) "$PY" -B pupils_f36c.py P01 0 --smoke ;;
    matrix)
        i=${SLURM_ARRAY_TASK_ID:?}
        printf -v held 'P%02d' "$((i%10+1))"
        "$PY" -B pupils_f36c.py "$held" "$((i/10))"
        ;;
    *) exit 2 ;;
esac
