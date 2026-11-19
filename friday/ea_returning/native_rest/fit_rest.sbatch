#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:09:00
#SBATCH --job-name=ni26-return-rest-fit
#SBATCH --output=rest-fit-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
here=${SLURM_SUBMIT_DIR:?}
cd "$here"
sha256sum --quiet -c fit_source.sha256
(cd "$base" && sha256sum --quiet -c "$here/fit_inputs.sha256")
export NI26_WS="$repo"
export PYTHONPATH="$repo/experiments:$repo/scripts:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
case "${1:?smoke or fit}" in
    smoke) exec "$repo/.venv/bin/python" -B fit_rest.py 0 --smoke ;;
    fit)
        test "$(sha256sum smoke/rest_s0.json | cut -d ' ' -f1)" = "${2:?adopted smoke hash}"
        exec "$repo/.venv/bin/python" -B fit_rest.py "${SLURM_ARRAY_TASK_ID:?fresh seed}"
        ;;
    *) exit 2 ;;
esac
