#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:02:00
#SBATCH --job-name=ni26-dea-matrix
#SBATCH --output=matrix-%A_%a.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
cd "$BASE/ni26_friday_ea_matrix_20261118"
sha256sum --quiet -c source.sha256
sha256sum --quiet -c archive.sha256
sha256sum --quiet -c parents.sha256
revision=c625ecc736a881836a25f85a741ea9147bf698bf
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export BENCHOPT_DATA_HOME="$REPO/data"
export PYTHONPATH="$PWD/$revision/experiments:$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
case "${1:?expected inputs, matrix or collect}" in
    inputs)
        "$REPO/.venv/bin/python" -B probe.py "$REPO" fbcnet_ea_tf inputs
        ;;
    matrix)
        test -s fbcnet_ea_tf_P01_seed0_inputs.json
        id=${SLURM_ARRAY_TASK_ID:?}
        printf -v person 'P%02d' "$((id%10+1))"
        "$REPO/.venv/bin/python" -B probe.py "$REPO" fbcnet_ea_tf matrix --held "$person" --seed "$((id/10))"
        ;;
    collect)
        "$REPO/.venv/bin/python" -B collect.py "$REPO"
        ;;
    *) exit 2 ;;
esac
