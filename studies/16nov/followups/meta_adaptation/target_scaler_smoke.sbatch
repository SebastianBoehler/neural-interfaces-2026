#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=ni26-s307-scaler-smoke
#SBATCH --output=logs/s307-scaler-smoke-%j.out
#SBATCH --error=logs/s307-scaler-smoke-%j.err
set -euo pipefail
cd "${S307_SCALER_SNAPSHOT:?}"
source studies/16nov/followups/meta_adaptation/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export S307_S305_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s305_assembler_recover_v1_20261116
export S301_ALL_REFERENCES=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
for f in source.sha256 pinned.sha256 target-src.sha256 meta-cpu-src.sha256; do sha256sum -c "$f" >/dev/null; done
(cd "$S307_CACHE" && sha256sum -c "$NI26_WS/cache.sha256" >/dev/null)
sha256sum -c meta.sha256 >/dev/null
"$PY" -B -I studies/16nov/followups/meta_adaptation/target_cpu.py P01
"$PY" -B -I studies/16nov/followups/meta_adaptation/target_check.py P01
