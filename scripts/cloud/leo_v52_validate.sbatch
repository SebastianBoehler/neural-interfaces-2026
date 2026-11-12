#!/bin/bash
#SBATCH --job-name=ni26-v52val
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=runs/diag/leo_v52_validate-%j.out
# v5.2 validation on one A100 (one node for all three steps, so the timing pair is fair): official replay of v5.2 and
# of v5.1b with COMPET_DEVICE=cuda, then scripts/check_v52.py on GPU (forced-off and person-switch passes included).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset PYTHONPATH; . .venv312/bin/activate
export BENCHOPT_DATA_HOME="$PWD/data" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi --query-gpu=name,driver_version --format=csv; hostname; date -u
python scripts/replay_package.py runs/sealed/cand_v52_pnets "runs/sealed/cand_v52_pnets_replay_leo_$SLURM_JOB_ID" cuda
python scripts/replay_package.py runs/sealed/cand_v51b_own "runs/sealed/cand_v51b_own_pair_leo_$SLURM_JOB_ID" cuda
python scripts/check_v52.py runs/sealed/cand_v52_pnets runs/sealed/cand_v51b_own --out "runs/sealed/cand_v52_check_gpu_$SLURM_JOB_ID.json" --device cuda
date -u; echo V52_VALIDATE_DONE
