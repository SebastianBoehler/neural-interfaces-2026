#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-302-throttle
#SBATCH --output=logs/302-throttle-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes302_20261116/repair
sha256sum -c repair-throttle.sha256
[[ "$(sacct -X -n -P -j 3114266 -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]]
wave=$(awk '$1 == "wave" {print $2}' runs/lane1_inputs/repair302.intent)
[[ "$wave" =~ ^[0-9]+$ ]] || { echo 'missing repair wave'; exit 1; }
if [[ -z "$(squeue -h -j "$wave" -o %T)" ]]; then
  [[ "$(sacct -X -n -P -j "$wave" -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]]
  echo 'wave already complete; no throttle change'
  exit 0
fi
running=$(squeue -h -u "$USER" -r -t RUNNING | wc -l)
(( running + 4 <= 32 )) || { echo "unsafe running capacity: $running"; exit 1; }
scontrol update JobId="$wave" ArrayTaskThrottle=6
scontrol show job "$wave" | grep -o 'ArrayTaskThrottle=[^ ]*' | head -1
printf 'wave %s raised from 2 to 6; running jobs were not altered\n' "$wave"
