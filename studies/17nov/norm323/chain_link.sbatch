#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-323-chain-link
#SBATCH --output=chain-link-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s323_interactions_20261117
sha256sum -c chain-interactions.sha256 >/dev/null
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I chain_link.py
