#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-return-eye-fit
#SBATCH --output=eye-fit-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
here=${SLURM_SUBMIT_DIR:?}
cd "$here"
sha256sum --quiet -c fit_source.sha256
(cd "$base" && sha256sum --quiet -c "$here/fit_inputs.sha256")
export NI26_WS="$repo"
export PYTHONPATH="$repo/experiments:$repo/scripts:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
people=(P04 P10)
slot=${SLURM_ARRAY_TASK_ID:?}
case "${1:?smoke or fit}" in
    smoke) exec "$repo/.venv/bin/python" -B fit_eye.py "${people[$slot]}" 0 --smoke ;;
    fit)
        "$repo/.venv/bin/python" -B verify_fit.py smoke
        exec "$repo/.venv/bin/python" -B fit_eye.py "${people[$((slot / 3))]}" "$((slot % 3))"
        ;;
    *) exit 2 ;;
esac
