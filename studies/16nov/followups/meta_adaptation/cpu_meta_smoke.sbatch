#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-s307-cpu-meta-smoke
#SBATCH --output=logs/s307-cpu-meta-smoke-%j.out
#SBATCH --error=logs/s307-cpu-meta-smoke-%j.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/cpu_meta_smoke
source studies/16nov/followups/meta_adaptation/env.sh
for f in source.sha256 pinned.sha256 cpu-src.sha256 meta-cpu-src.sha256; do sha256sum -c "$f" >/dev/null; done
sha256sum -c pilot-copy.sha256 >/dev/null
"$PY" -B -I studies/16nov/followups/meta_adaptation/meta_train_cpu.py P01 scalar 0 smoke
"$PY" -B -I studies/16nov/followups/meta_adaptation/meta_train_cpu.py P01 blocks 0 smoke
