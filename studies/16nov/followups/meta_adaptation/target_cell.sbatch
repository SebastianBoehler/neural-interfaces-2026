#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=ni26-s307-target
#SBATCH --output=logs/s307-target-%A_%a.out
#SBATCH --error=logs/s307-target-%A_%a.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/target_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
export S307_S305_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s305_assembler_recover_v1_20261116
for f in source.sha256 pinned.sha256 target-src.sha256 meta-cpu-src.sha256; do sha256sum -c "$f" >/dev/null; done
(cd "$S307_CACHE" && sha256sum -c "$NI26_WS/cache.sha256" >/dev/null)
if [[ ${1:-} == smoke ]]; then held=P01; else printf -v held 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"; fi
awk -v prefix="runs/s307/meta/${held}_" '$2 ~ "^" prefix {print}' meta.sha256 > "logs/meta-${held}.sha256"
test "$(wc -l < "logs/meta-${held}.sha256")" = 4
sha256sum -c "logs/meta-${held}.sha256" >/dev/null
"$PY" -B -I studies/16nov/followups/meta_adaptation/target_cpu.py "$held"
"$PY" -B -I studies/16nov/followups/meta_adaptation/target_check.py "$held"
set -o noclobber
printf 'CPU %s %s\n' "$(sha256sum "runs/s307/target/$held.npz" | cut -d' ' -f1)" "$(sha256sum target-src.sha256 | cut -d' ' -f1)" > "runs/s307/target/$held.cpu-route"
