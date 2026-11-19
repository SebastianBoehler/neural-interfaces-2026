#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 6
#SBATCH -J ni26-score-f5
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
work=$PWD
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$prior:$repo/experiments
sha256sum -c --status external_source.sha256
(cd "$prior" && sha256sum -c --status "$work/external_parents.sha256")
case $1 in
    verify)
        (cd "$prior" && sha256sum -c --status inputs.sha256 && sha256sum -c --status members.sha256)
        printf '%s  %s\n' 9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed "$prior/v66_cp1_cap250.zip" | sha256sum -c --status
        [[ $(git -C "$repo/2026-competition" rev-parse HEAD) == a66bc23d1a0e686b895c093a642b45c74970758b ]]
        (cd "$base/ni26_s318_coral_fbl_20261117" && "$repo/.venv/bin/python" replay/check.py) > replay.log
        grep -Fx OK replay.log
        ;;
    score) "$repo/.venv/bin/python" external_score.py ;;
    *) exit 2 ;;
esac
