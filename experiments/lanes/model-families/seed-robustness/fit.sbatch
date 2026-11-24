#!/bin/bash
#SBATCH --account=gehler
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-laneT-fit
#SBATCH --output=%x-%j.out
# usage (cd to the lane T release first): sbatch [--partition=..] [--gres=gpu:..] fit.sbatch ORDER ARM FAMILY [smoke]
# builds ./arms/ORDER_ARM as a copy of the pinned matched-fit release (code only), adds ttt_core/fit_entry, runs one family.
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
CAMP=$BASE/ni26_friday_causal_rebuild_20261118
HERE=${SLURM_SUBMIT_DIR:?}
order=${1:?}; arm=${2:?}; family=${3:?}; mode=${4:-production}
case "$order" in
    full0) src=$CAMP/full0_common_matched_fit_v1; bridge_dir=$CAMP/full0_common_bridge_v1; inputs=full0_common_inputs.sha256 ;;
    within0) src=$CAMP/within0_matched_refit_shared_v1; bridge_dir=$CAMP/within0_shared_bridge_v1; inputs=within0_warm_inputs.sha256 ;;
    *) exit 2 ;;
esac
dir=$HERE/arms/${order}_${arm}
if [[ ! -d $dir ]]; then
    mkdir -p "$dir.tmp$$"
    (cd "$src" && cp *.py *.sha256 *.sbatch "$dir.tmp$$/")
    cp "$HERE/ttt_core.py" "$HERE/fit_entry.py" "$dir.tmp$$/"
    mkdir -p "$dir.tmp$$/members" "$dir.tmp$$/smoke"
    mv "$dir.tmp$$" "$dir" 2>/dev/null || rm -rf "$dir.tmp$$"
fi
cd "$dir"
sha256sum --quiet -c source.sha256
(cd "$BASE" && sha256sum --quiet -c "$dir/$inputs")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$dir:$REPO/experiments:$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
bridge=$(sha256sum "$bridge_dir/status.json" | cut -d ' ' -f 1)
[[ $family == sinc ]] && export CUBLAS_WORKSPACE_CONFIG=:4096:8
flag=""; [[ $mode == smoke ]] && flag="--smoke"
echo "laneT fit $order $arm $family $mode on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
start=$(date +%s)
"$REPO/.venv/bin/python" -B fit_entry.py "$order" "$arm" "$family" "$bridge" $flag
echo "laneT seconds $(( $(date +%s) - start ))"
