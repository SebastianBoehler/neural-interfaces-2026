#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-79%2
#SBATCH --output=logs/production-%A_%a.out
#SBATCH --error=logs/production-%A_%a.err
set -euo pipefail
source studies/16nov/lane5/cluster_env.sh
export LANE5_DEVICE=1
check_inputs
require_gate cpu_parity
require_gate gpu_parity_smoke
"$PYTHON" -B -I studies/16nov/lane5/run.py "$SLURM_ARRAY_TASK_ID"
check_inputs
