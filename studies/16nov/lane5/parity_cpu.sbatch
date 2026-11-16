#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=logs/parity-cpu-%j.out
#SBATCH --error=logs/parity-cpu-%j.err
set -euo pipefail
source studies/16nov/lane5/cluster_env.sh
export LANE5_DEVICE=0
check_inputs
export LANE5_LIBRARY="$LANE5_LIB"
"$PYTHON" -B -I -m unittest discover -s studies/16nov/lane5/native -p test_contract.py
"$PYTHON" -B -I studies/16nov/lane5/native/test_publication.py
"$PYTHON" -B -I studies/16nov/lane5/parity.py
check_inputs
write_gate cpu_parity
