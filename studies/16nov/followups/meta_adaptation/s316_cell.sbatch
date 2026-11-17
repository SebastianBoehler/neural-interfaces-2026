#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s316-target
#SBATCH --output=logs/s316-target-%A_%a.out
#SBATCH --error=logs/s316-target-%A_%a.err
set -euo pipefail
cd "${S316_TARGET_WS:?}"
source studies/16nov/followups/meta_adaptation/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export S307_S305_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s305_assembler_recover_v1_20261116
export S301_ALL_REFERENCES=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
for f in source.sha256 pinned.sha256 target-src.sha256 meta-cpu-src.sha256 meta.sha256; do sha256sum -c "$f" >/dev/null; done
(cd "$S307_CACHE" && sha256sum -c "$NI26_WS/cache.sha256" >/dev/null)
if [[ ${1:-} == pilot ]]; then held=P09; else printf -v held 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"; fi
"$PY" -B -I studies/16nov/followups/meta_adaptation/s316_target.py "$held"
"$PY" -B -I studies/16nov/followups/meta_adaptation/target_check.py "$held"
set -o noclobber
printf 'CPU %s %s\n' "$(sha256sum "runs/s307/target/$held.npz" | cut -d' ' -f1)" "$(sha256sum target-src.sha256 | cut -d' ' -f1)" > "runs/s307/target/$held.cpu-route"
