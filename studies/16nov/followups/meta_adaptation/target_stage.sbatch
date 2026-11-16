#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s307-target-stage
#SBATCH --output=logs/s307-target-stage-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/target_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
for f in source.sha256 pinned.sha256 target-src.sha256 meta-cpu-src.sha256; do sha256sum -c "$f" >/dev/null; done
source_root=/mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production_cpu_v2
cmp meta-cpu-src.sha256 "$source_root/meta-cpu-src.sha256"
test -f /mnt/lustre/work/gehler/gfh527/ni26_s305_assembler_recover_v1_20261116/runs/s305_converged/manifest.json
test ! -e runs/s307/meta
mkdir -p runs/s307/meta
cp "$source_root"/runs/s307/meta/*.npz "$source_root"/runs/s307/meta/*.cpu-route runs/s307/meta/
"$PY" -B -I studies/16nov/followups/meta_adaptation/verify_meta.py
set -o noclobber
printf 'stage %s\n' "$SLURM_JOB_ID" > target-stage.intent
set +o noclobber
pilot=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" studies/16nov/followups/meta_adaptation/target_cell.sbatch smoke)
printf 'pilot %s\n' "$pilot" >> target-stage.intent
gate=$(sbatch --parsable --dependency=afterok:"$pilot" studies/16nov/followups/meta_adaptation/target_release.sbatch)
printf 'gate %s\n' "$gate" >> target-stage.intent
