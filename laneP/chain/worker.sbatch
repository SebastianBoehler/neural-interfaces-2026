#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-laneP-chain
#SBATCH --output=laneP-%x-%j.out
# usage (cd to the lane P chain dir): sbatch worker.sbatch ORDER smoke|complete "cbbank_ORDER_s*.npz" | worker.sbatch score
# helper modules (parents22, matrix22, cross22, final22) come from the final-stack release, hash-checked below
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
camp=$base/ni26_friday_causal_rebuild_20261118
root=$base/ni26_s318_coral_fbl_20261117
repo=$base/neural-interfaces-2026
c18=$camp/fc18_necessity_v2
c19=$camp/fc19_csp_gate_v2
c20=$camp/fc20_outer_v1
f32=$camp/f32c_v2
work=${SLURM_SUBMIT_DIR:?}
fs=$base/ni26_final_stack_20261120
export LANET_BANKS=${LANET_BANKS:-$base/ni26_laneT_20261120/laneT/arms}
export LANEP_SEG=${LANEP_SEG:-$base/ni26_laneB5_20261120/laneP/seg}
cd "$work"
sha256sum --check --status source.sha256
(cd "$fs" && sha256sum --check --status source.sha256)
mode=${1:?}
case "$mode" in
    within0|score)
        parent=$camp/within0_terminal_shared_v1
        parent_sha=ad7f624831d9e25ab636a7096a07a6e64999bdf47e7dcea3994e4210b84d8326 ;;
    full0)
        parent=$camp/full0_common_terminal_v1
        parent_sha=74527664de7163c98b4f633c5f926fdec471b15cc656baec7c5810b58b9d335a ;;
    *) exit 2 ;;
esac
verify() ( cd "$1"; echo "$2  source.sha256" | sha256sum --check --status; sha256sum --check --status source.sha256 )
verify "$parent" "$parent_sha"
verify "$c18" dfbd2c851dc4fbc276593ec521db7155184487d73ec1338b6e80949344e7e354
verify "$c19" 0154ffdc8358f6a7f7865f855f70203400cda95ca5b7924478f57467a1fc8cd7
(cd "$c20" && echo '1319f8c515cba34eb9c7347c7b94a584ae2f33b3225c01ecfb623c9f60506f5b  consumer.sha256' | sha256sum --check --status
 sha256sum --check --status consumer.sha256)
(cd "$f32" && echo '38c145d54b757d9d1feee6408b32c0ed0bb7c2bf1c5d71611b86ee27ce84d518  consumer.sha256' | sha256sum --check --status -
 sha256sum --check --status consumer.sha256
 echo '4a7f749f7074665c046bac45e4d8eb70b88dd3dad7b68f2153ede473076706a6  pupil.sha256' | sha256sum --check --status -
 sha256sum --check --status pupil.sha256)
grep -Fx OK "$base/ni26_cross411b_20261118/replay-proof.txt"
(cd "$root"; sha256sum --check --status "$parent/owner.sha256"; sha256sum --check --status "$parent/recognition.sha256"
 sha256sum --check --status "$parent/complete_terminal_inputs.sha256")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$work:$fs:$c20:$parent:$c18:$c19:$f32"
if [[ $mode == score ]]; then
    exec "$repo/.venv/bin/python" -B score_p.py
fi
exec "$repo/.venv/bin/python" -B final_p.py "$mode" "${2:?}" "${3:?}"
