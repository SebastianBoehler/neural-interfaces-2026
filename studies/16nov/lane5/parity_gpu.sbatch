#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/parity-gpu-%j.out
#SBATCH --error=logs/parity-gpu-%j.err
set -euo pipefail
source studies/16nov/lane5/cluster_env.sh
export LANE5_DEVICE=1
check_inputs
require_gate cpu_parity
"$PYTHON" -B -I studies/16nov/lane5/parity.py
"$PYTHON" -B -I studies/16nov/lane5/smoke.py
check_inputs
write_gate gpu_parity_smoke
