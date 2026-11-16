#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=logs/collect-%j.out
#SBATCH --error=logs/collect-%j.err
set -euo pipefail
source studies/16nov/lane5/cluster_env.sh
check_inputs
require_gate cpu_parity
require_gate gpu_parity_smoke
"$PYTHON" -B -I studies/16nov/lane5/score.py
check_inputs
