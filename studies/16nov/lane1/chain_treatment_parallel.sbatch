#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-l1-chain-v2
#SBATCH --output=logs/chain-v2-%A_%a.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c parallel-src.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
if [[ $1 == coral ]]; then
    arms=(C3A C3S C10A C10S)
elif [[ $1 == eeg ]]; then
    arms=(zero_eegn E D)
else
    exit 2
fi
arm=${arms[$((SLURM_ARRAY_TASK_ID / 8))]}
order=${orders[$((SLURM_ARRAY_TASK_ID % 8))]}
case "$arm" in
    C3A) ADD=fb_coral3; SWAP=none; ZERO=0 ;;
    C3S) ADD=none; SWAP=fb_coral3; ZERO=0 ;;
    C10A) ADD=fb_coral10; SWAP=none; ZERO=0 ;;
    C10S) ADD=none; SWAP=fb_coral10; ZERO=0 ;;
    E) ADD=eegn; SWAP=none; ZERO=0 ;;
    D) ADD=eegn_dist2; SWAP=none; ZERO=0 ;;
    zero_eegn) ADD=eegn; SWAP=none; ZERO=1 ;;
esac
export NM_NAME="$arm" ADD SWAP ZERO NM_SEEDS=0,1,2,3,4,5
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/14nov/newmember/run_s_v66.py "$order"
"$PY" -B -I studies/16nov/lane1/parallel_cp.py "$arm" "$order"
