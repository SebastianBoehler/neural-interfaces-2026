#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-eye-cal-fit
#SBATCH --output=eye-cal-fit-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c fit_source.sha256
(cd "$base" && sha256sum --quiet -c "$work/fit_inputs.sha256")
export NI26_WS="$repo"
export PYTHONPATH="$repo/experiments:$repo/scripts:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python="$repo/.venv/bin/python"
case "${1:?smoke or fit}" in
    smoke) exec "$python" -B joint_eye.py 0 --smoke ;;
    fit)
        "$python" -B verify_joint.py
        exec "$python" -B joint_eye.py "${SLURM_ARRAY_TASK_ID:?}"
        ;;
    *) exit 2 ;;
esac
