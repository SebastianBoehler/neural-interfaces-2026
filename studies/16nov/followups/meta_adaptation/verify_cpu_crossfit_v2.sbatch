#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-s307-verify
#SBATCH --output=logs/s307-verify-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
sha256sum -c source.sha256 >/dev/null
sha256sum -c dispatch.sha256 >/dev/null
sha256sum -c cpu-src.sha256 >/dev/null
"$PY" -B -I studies/16nov/followups/meta_adaptation/verify_cpu_crossfit.py
