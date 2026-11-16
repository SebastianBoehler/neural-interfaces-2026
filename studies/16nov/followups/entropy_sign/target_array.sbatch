#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-s310-target
#SBATCH --output=logs/s310-target-%A_%a.out
#SBATCH --error=logs/s310-target-%A_%a.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s310_entropy_sign_20261116/production_v2
source studies/16nov/followups/meta_adaptation/env.sh
export S310_S305=/mnt/lustre/work/gehler/gfh527/ni26_s305_converged_v2_20261116
export S310_S305_ASSEMBLER=/mnt/lustre/work/gehler/gfh527/ni26_s305_assembler_recover_v1_20261116
export S310_REF=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
for f in source.sha256 pinned.sha256 cpu-src.sha256 s310-src.sha256 pilot-pinned.sha256; do sha256sum -c "$f" >/dev/null; done
(cd "$S307_CACHE" && sha256sum -c "$NI26_WS/cache.sha256" >/dev/null)
printf -v held 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"
"$PY" -B -I studies/16nov/followups/entropy_sign/verify_shard.py "$held"
"$PY" -B -I studies/16nov/followups/entropy_sign/produce.py "$held"
path="runs/s310/target/$held.npz"
set -o noclobber
printf 'CPU %s %s\n' "$(sha256sum "$path" | cut -d' ' -f1)" "$(sha256sum s310-src.sha256 | cut -d' ' -f1)" > "${path%.npz}.cpu-route"
