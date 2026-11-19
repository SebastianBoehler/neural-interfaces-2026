#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 4
#SBATCH -J ni26-f16-distill
#SBATCH -o fit-%A_%a.out
set -euo pipefail
cd ~/ni26/friday_s1c_distill_20261118
f14=$(cd ../friday_s1c_training_20261118 && pwd -P)
f15=$(cd ../friday_s1c_skip_20261118 && pwd -P)
sha256sum --quiet -c actor.sha256
(cd "$f14" && sha256sum --quiet -c actor.sha256)
(cd "$f15" && sha256sum --quiet -c actor.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$f14:$f14/original:$f14/original/experiments"
"$f14/runtime/bin/python" -m pip freeze --all | LC_ALL=C sort | cmp -s "$f14/runtime.lock" -
case "${1:?expected smoke or matrix}" in
    smoke) "$f14/runtime/bin/python" -B produce_distill.py P01 0 --smoke ;;
    matrix)
        id=${SLURM_ARRAY_TASK_ID:?}
        printf -v held 'P%02d' "$((id%10+1))"
        "$f14/runtime/bin/python" -B produce_distill.py "$held" "$((id/10))"
        ;;
    *) exit 2 ;;
esac
