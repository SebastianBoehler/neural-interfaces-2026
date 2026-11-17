#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-cpucell
set -euo pipefail
CPUROOT=/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2
LOCK=/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_release_20261117
(cd "$LOCK" && sha256sum -c release-source.sha256)
cd "$CPUROOT"
/usr/bin/time -o probe_timing.json -f '{"elapsed":%e,"maxrss_kb":%M}' bash work_cpu.sbatch probe
