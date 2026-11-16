#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s308-member
#SBATCH --output=logs/s308-member-%A_%a.out
#SBATCH --error=logs/s308-member-%A_%a.err
set -euo pipefail
cd "${S308_CHAIN:?}"
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export S301_MANIFEST="$PWD/runs/s301_native/manifest.json"
export S301_MANIFEST_SHA="$(sha256sum "$S301_MANIFEST" | cut -d' ' -f1)"
unset PYTHONPATH
sha256sum -c s295-source.sha256 >/dev/null
sha256sum -c chain.sha256 >/dev/null
test "$(sha256sum chain.sha256 | cut -d' ' -f1)" = "$S308_CHAIN_SHA"
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
if [[ ${1:-} == zero ]]; then
  arm=Rzero; order=${orders[$SLURM_ARRAY_TASK_ID]}
elif [[ ${1:-} == treatment ]]; then
  arms=(Slinear0 Slinear1 Stanh0 Stanh1 Rlinear0 Rlinear1 Rquadratic0 Rquadratic1 Rfourier0 Rfourier1)
  arm=${arms[$((SLURM_ARRAY_TASK_ID / 8))]}; order=${orders[$((SLURM_ARRAY_TASK_ID % 8))]}
else
  exit 2
fi
srun --cpu-bind=cores bash studies/16nov/followups/strong_heads/member_step.sh "$arm" "$order"
