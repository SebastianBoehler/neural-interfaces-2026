#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani,2080-galvani,a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-s307-crossfit
#SBATCH --output=logs/s307-crossfit-%A_%a.out
#SBATCH --error=logs/s307-crossfit-%A_%a.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production
source studies/16nov/followups/meta_adaptation/env.sh
sha256sum -c source.sha256 >/dev/null
sha256sum -c pinned.sha256 >/dev/null
sha256sum -c dispatch.sha256 >/dev/null
(cd "$S307_CACHE" && sha256sum -c "$NI26_WS/cache.sha256" >/dev/null)
n=${SLURM_ARRAY_TASK_ID:?}; (( n >= 0 && n < 90 ))
o=$((n / 9)); s=$((n % 9)); ((s >= o)) && s=$((s+1))
printf -v outer 'P%02d' "$((o+1))"; printf -v inner 'P%02d' "$((s+1))"
"$PY" -B -I studies/16nov/followups/meta_adaptation/crossfit.py "$outer" "$inner"
