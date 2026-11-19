#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 6
#SBATCH -J ni26-f37c-pupils
#SBATCH -o pupils-%A_%a.out
set -euo pipefail
b=/mnt/lustre/work/gehler/gfh527
o=$b/ni26_s323_normsplit_20261117
f14=$b/ni26_friday_s1c_training_20261118
f16=$b/ni26_friday_s1c_distill_20261118
f17=$b/ni26_friday_s1c_phase_20261118
f37=$b/ni26_friday_s1c_spectral_shift_20261119
c20=$b/ni26_friday_causal_rebuild_20261118/fc20_pupils_v2
f32=$b/ni26_friday_causal_rebuild_20261118/f32c_v2
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
verify "$f37" 0575236e532b5858a28c0e510d8fda3030cc5f4584648aa657642347e6f833bd actor.sha256
verify "$f37" 70deb2a3c25e1a2a569afd435aea726e9722ad5e442440dfa00fb7e79a90e13d score.sha256
verify "$c20" 5925163a6331fc062a51f80f8262c05f7c12f069cfdba86d9e2cddf1f57d349c source.sha256
verify "$f32" 4a7f749f7074665c046bac45e4d8eb70b88dd3dad7b68f2153ede473076706a6 pupil.sha256
export PYTHONPATH="$f37:$o:$o/experiments:$o/studies/14nov/newmember:$f14:$f16:$f17:$c20:$f32"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=$b/neural-interfaces-2026/.venv/bin/python
case "${1:?expected smoke or matrix}" in
    smoke) "$PY" -B pupils_f37c.py P01 0 --smoke ;;
    matrix)
        i=${SLURM_ARRAY_TASK_ID:?}
        printf -v held 'P%02d' "$((i%10+1))"
        "$PY" -B pupils_f37c.py "$held" "$((i/10))"
        ;;
    *) exit 2 ;;
esac
