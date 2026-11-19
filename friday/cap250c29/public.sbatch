#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:12:00
#SBATCH --job-name=ni26-c29-public
#SBATCH --output=c29-public-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
camp=$base/ni26_friday_causal_rebuild_20261118
root=$base/ni26_s318_coral_fbl_20261117
repo=$base/neural-interfaces-2026
c18=$camp/fc18_necessity_v2
c24=$base/ni26_friday_c24_temporal_csp_public_v2_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
sha256sum --check --status public.sha256
sha256sum --check --status source-fit.sha256
(cd "$c24" && echo 'b0564221e6e2b77e36fdc7114a1b5031d836ded05be50a55d78775fa652d938e  public.sha256' | sha256sum --check --status
 sha256sum --check --status public.sha256)
grep -Fx OK "$base/ni26_cross411b_20261118/replay-proof.txt"
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
(
    cd "$parent"
    echo "$parent_sha  source.sha256" | sha256sum --check --status
    sha256sum --check --status source.sha256
)
(
    cd "$c18"
    echo "dfbd2c851dc4fbc276593ec521db7155184487d73ec1338b6e80949344e7e354  source.sha256" | sha256sum --check --status
    sha256sum --check --status source.sha256
)
(
    cd "$root"
    sha256sum --check --status "$parent/owner.sha256"
    sha256sum --check --status "$parent/recognition.sha256"
    sha256sum --check --status "$parent/complete_terminal_inputs.sha256"
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$work:$c24:$parent:$c18"
if [[ $mode == score ]]; then
    exec "$repo/.venv/bin/python" -B scorepublic29.py
fi
exec "$repo/.venv/bin/python" -B public29.py "$mode"
