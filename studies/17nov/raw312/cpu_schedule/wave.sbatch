#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-cpushort
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_schedule_20261117
sha256sum -c schedule-source.sha256
ROOT=/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2
(cd "$ROOT" && sha256sum -c source.sha256 >/dev/null)
export NI26_WS="$ROOT"
"$ROOT/.venv/bin/python" -B wave.py "${1:?}"
