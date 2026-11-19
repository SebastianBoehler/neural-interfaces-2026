#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-f35c-whole
#SBATCH --output=whole-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
camp=$base/ni26_friday_causal_rebuild_20261118
root=$base/ni26_s318_coral_fbl_20261117
repo=$base/neural-interfaces-2026
c18=$camp/fc18_necessity_v2
c19=$camp/fc19_csp_gate_v2
c20=$camp/fc20_outer_v1
f25c=$camp/f25c_v1
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
for manifest in pupil whole; do
    sha256sum --check --status "$manifest.sha256"
done
mode=${1:?}
case "$mode" in
    within0|score)
        parent=$camp/within0_terminal_shared_v1
        parent_sha=ad7f624831d9e25ab636a7096a07a6e64999bdf47e7dcea3994e4210b84d8326
        ;;
    full0)
        parent=$camp/full0_common_terminal_v1
        parent_sha=74527664de7163c98b4f633c5f926fdec471b15cc656baec7c5810b58b9d335a
        ;;
    *) exit 2 ;;
esac
verify() (
    cd "$1"
    printf '%s  %s\n' "$2" "$3" | sha256sum --check --status
    sha256sum --check --status "$3"
)
verify "$parent" "$parent_sha" source.sha256
verify "$c18" dfbd2c851dc4fbc276593ec521db7155184487d73ec1338b6e80949344e7e354 source.sha256
verify "$c19" 0154ffdc8358f6a7f7865f855f70203400cda95ca5b7924478f57467a1fc8cd7 source.sha256
verify "$c20" 1319f8c515cba34eb9c7347c7b94a584ae2f33b3225c01ecfb623c9f60506f5b consumer.sha256
verify "$f25c" 2f928083d9163a9e9530a746eada4438fc134f76b2f5122139c02fb37894caa7 consumer.sha256
verify "$f25c" 094fdadfddbe1c5705e7686c84f00e46bd6c782ceb40c32e715f2bcb0e003301 scoring.sha256
(
    cd "$root"
    sha256sum --check --status "$parent/owner.sha256"
    sha256sum --check --status "$parent/recognition.sha256"
    sha256sum --check --status "$parent/complete_terminal_inputs.sha256"
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$work:$f25c:$c20:$parent:$c18:$c19"
if [[ $mode == score ]]; then
    exec "$repo/.venv/bin/python" -B score_f35c.py
fi
exec "$repo/.venv/bin/python" -B whole_f35c.py "$mode" "${2:?}"
