#!/bin/bash
#SBATCH --job-name=ni26-policy350-collect
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:04:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$py" collect_gpu_guard.py preflight > collect_preflight.log
work="$PWD"
cd /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
"$py" replay/check.py > "$work/replay_check.txt"
cd "$work"
"$py" collect_gpu_guard.py collect
