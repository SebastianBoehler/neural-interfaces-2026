#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-304-chain
#SBATCH --output=logs/304-chain-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s304_coral_chain_20261116
sha256sum -c chain304-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c coral304-chain-src.sha256
sha256sum -c runs/coral304_inputs/registration.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
arms=(PLA PLS UCA UCS CCA10 CCS10 CCA30 CCS30 zero304)
run_cell() {
 local arm=$1 order=$2
 case "$arm" in
  PLA) ADD=fb304; SWAP=none; ZERO=0;; PLS) ADD=none; SWAP=fb304; ZERO=0;;
  UCA) ADD=fb_uc304; SWAP=none; ZERO=0;; UCS) ADD=none; SWAP=fb_uc304; ZERO=0;;
  CCA10) ADD=fb_cc10; SWAP=none; ZERO=0;; CCS10) ADD=none; SWAP=fb_cc10; ZERO=0;;
  CCA30) ADD=fb_cc30; SWAP=none; ZERO=0;; CCS30) ADD=none; SWAP=fb_cc30; ZERO=0;;
  zero304) ADD=fb304; SWAP=none; ZERO=1;; *) return 2;;
 esac
 export NM_NAME="$arm" ADD SWAP ZERO NM_SEEDS=0,1,2,3,4,5
 "$PY" -B -I studies/14nov/newmember/run_s_v66.py "$order"
 "$PY" -B -I studies/16nov/followups/coral_repair/chain/cp_parallel.py "$arm" "$order"
}
if [[ ${1:-} == smoke ]]; then
 for arm in zero304 PLA CCS10; do run_cell "$arm" within0; done
else
 id=${SLURM_ARRAY_TASK_ID:?}; run_cell "${arms[$((id / 8))]}" "${orders[$((id % 8))]}"
fi
