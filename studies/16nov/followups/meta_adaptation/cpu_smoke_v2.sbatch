#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-s307-cpu-smoke
#SBATCH --output=logs/s307-cpu-smoke-%j.out
#SBATCH --error=logs/s307-cpu-smoke-%j.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/cpu_smoke_v2
source studies/16nov/followups/meta_adaptation/env.sh
sha256sum -c source.sha256 >/dev/null
sha256sum -c pinned.sha256 >/dev/null
sha256sum -c cpu-src.sha256 >/dev/null
(cd "$S307_CACHE" && sha256sum -c "$NI26_WS/cache.sha256" >/dev/null)
"$PY" -B -I studies/16nov/followups/meta_adaptation/cpu_crossfit.py P01 P02 smoke
"$PY" -B -I studies/16nov/followups/meta_adaptation/cpu_parity.py
