#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-fat-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-shallow-fresh
#SBATCH --output=fresh-%A_%a.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
PARENT=$BASE/ni26_friday_ea_matrix_20261118
cd "$BASE/ni26_friday_ea_shallow_fresh_20261118"
sha256sum --quiet -c source.sha256
sha256sum --quiet -c inputs.sha256
(
    cd "$PARENT"
    echo "a01929b52e7d7ba3ecaa001785c2f3a0811fd6dbb0564fde5070d0418af35713  source.sha256" | sha256sum --quiet -c -
    sha256sum --quiet -c source.sha256
)
revision=c625ecc736a881836a25f85a741ea9147bf698bf
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 BENCHOPT_DATA_HOME="$REPO/data"
export PYTHONPATH="$PARENT:$PARENT/$revision/experiments:$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
case "${1:?expected smoke, matrix or collect}" in
    smoke)
        "$REPO/.venv/bin/python" -B new_fit.py "$REPO" P10 0 --smoke
        ;;
    matrix)
        test -s P10_seed0_receipt.json
        id=${SLURM_ARRAY_TASK_ID:?}
        printf -v person 'P%02d' "$((id%10+1))"
        "$REPO/.venv/bin/python" -B new_fit.py "$REPO" "$person" "$((id/10))"
        ;;
    collect)
        "$REPO/.venv/bin/python" -B new_collect.py "$REPO"
        ;;
    *) exit 2 ;;
esac
