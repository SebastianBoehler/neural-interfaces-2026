#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-s307-cpu-crossfit
#SBATCH --output=logs/s307-cpu-crossfit-%A_%a.out
#SBATCH --error=logs/s307-cpu-crossfit-%A_%a.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
for manifest in source.sha256 pinned.sha256 cpu-src.sha256 dispatch.sha256; do sha256sum -c "$manifest" >/dev/null; done
(cd "$S307_CACHE" && sha256sum -c "$NI26_WS/cache.sha256" >/dev/null)
n=${SLURM_ARRAY_TASK_ID:?}; (( n >= 0 && n < 90 )); o=$((n / 9)); s=$((n % 9)); ((s >= o)) && s=$((s+1))
printf -v outer 'P%02d' "$((o+1))"; printf -v inner 'P%02d' "$((s+1))"
"$PY" -B -I studies/16nov/followups/meta_adaptation/cpu_crossfit.py "$outer" "$inner"
path="runs/s307/crossfit/${outer}_${inner}.npz"
set -o noclobber
printf 'CPU %s %s\n' "$(sha256sum "$path" | cut -d' ' -f1)" "$(sha256sum cpu-src.sha256 | cut -d' ' -f1)" > "${path%.npz}.cpu-route"
