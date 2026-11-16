#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s307-target-release
#SBATCH --output=logs/s307-target-release-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/target_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
for f in source.sha256 pinned.sha256 target-src.sha256 meta-cpu-src.sha256; do sha256sum -c "$f" >/dev/null; done
sha256sum -c meta.sha256 >/dev/null
"$PY" -B -I studies/16nov/followups/meta_adaptation/target_verify_pilot.py
active=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
if (( active + 13 > 272 )); then
 retry=$(sbatch --parsable --begin=now+10minutes studies/16nov/followups/meta_adaptation/target_release.sbatch)
 printf 'capacity %s; retry %s\n' "$active" "$retry" >> target-release-retries.log
 exit 0
fi
set -o noclobber
printf 'release %s\n' "$SLURM_JOB_ID" > target-release.intent
set +o noclobber
array=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=1-9%2 studies/16nov/followups/meta_adaptation/target_cell.sbatch)
printf 'target %s\n' "$array" >> target-release.intent
assembly=$(sbatch --parsable --dependency=afterok:"$array" studies/16nov/followups/meta_adaptation/target_assemble.sbatch)
printf 'assembly %s\n' "$assembly" >> target-release.intent
