#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-fat-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-ear-fit
#SBATCH --output=returning-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
parent=$base/ni26_friday_ea_matrix_20261118
shallow=$base/ni26_friday_ea_shallow_fresh_20261118
revision=c625ecc736a881836a25f85a741ea9147bf698bf
sha256sum --quiet -c source.sha256
sha256sum --quiet -c inputs.sha256
(cd "$parent" && sha256sum --quiet -c source.sha256)
(cd "$shallow" && sha256sum --quiet -c source.sha256)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 BENCHOPT_DATA_HOME="$repo/data"
export PYTHONPATH="$shallow:$parent:$parent/$revision/experiments:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
id=${SLURM_ARRAY_TASK_ID:?}
groups=(P04 P10)
person=${groups[$((id%2))]}
if ((id<10)); then
    family=fbcnet_ea_tf
    seed=$((id/2))
else
    family=shallow_ea
    seed=$(((id-10)/2))
fi
opts=()
if [[ $1 == smoke ]]; then
    [[ $id == 1 || $id == 11 ]]
    opts=(--smoke)
else
    [[ $1 == produce && $id != 1 && $id != 11 ]]
    sha256sum --quiet -c smoke.sha256
fi
"$repo/.venv/bin/python" -B returning_fit.py "$repo" "$family" "$person" "$seed" "${opts[@]}"
