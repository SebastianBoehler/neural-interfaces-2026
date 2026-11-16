#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s307-chain
#SBATCH --output=logs/s307-chain-%A_%a.out
#SBATCH --error=logs/s307-chain-%A_%a.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/chain_v2
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export S301_MANIFEST="$PWD/runs/s301_native/manifest.json" S301_MANIFEST_SHA="$(sha256sum runs/s301_native/manifest.json | cut -d' ' -f1)"
for f in s295-source.sha256 chain.sha256 s307-chain.sha256; do sha256sum -c "$f" >/dev/null; done
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
if [[ ${1:-} == zero ]]; then arm=Rzero; order=${orders[${SLURM_ARRAY_TASK_ID:?}]};
elif [[ ${1:-} == smoke ]]; then arm=Anchor; order=within0;
elif [[ ${1:-} == treatment ]]; then arms=(Anchor Fixed Scalar Blocks Scalar0 Scalar1 Blocks0 Blocks1); n=${SLURM_ARRAY_TASK_ID:?}; arm=${arms[$((n/8))]}; order=${orders[$((n%8))]};
else exit 2; fi
srun --cpu-bind=cores bash studies/16nov/followups/meta_adaptation/chain_step.sh "$arm" "$order"
