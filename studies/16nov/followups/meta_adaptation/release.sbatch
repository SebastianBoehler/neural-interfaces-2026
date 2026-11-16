#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s307-release
#SBATCH --output=logs/s307-release-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production
source studies/16nov/followups/meta_adaptation/env.sh
sha256sum -c source.sha256 >/dev/null
sha256sum -c pinned.sha256 >/dev/null
sha256sum -c dispatch.sha256 >/dev/null
cmp source.sha256 ../smoke/source.sha256
cmp pinned.sha256 ../smoke/pinned.sha256
cmp cache.sha256 ../smoke/cache.sha256
test "$(sacct -X -n -P -j 3115164 -o State,ExitCode | head -1)" = 'COMPLETED|0:0'
sacct -X -n -P -j 3115164 -o JobID,Elapsed,State,ExitCode
"$PY" -B -I studies/16nov/followups/meta_adaptation/verify_pilot.py
active=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
if (( active + 102 > 272 )); then
  retry=$(sbatch --parsable --begin=now+10minutes studies/16nov/followups/meta_adaptation/release.sbatch)
  printf 'capacity %s; retry %s\n' "$active" "$retry" >> release-retries.log
  exit 0
fi
set -o noclobber
printf 'release %s\n' "$SLURM_JOB_ID" > release.intent
set +o noclobber
crossfit=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=0-89%2 studies/16nov/followups/meta_adaptation/crossfit_array.sbatch)
printf 'crossfit %s\n' "$crossfit" >> release.intent
verify=$(sbatch --parsable --dependency=afterok:"$crossfit" studies/16nov/followups/meta_adaptation/verify_crossfit.sbatch)
printf 'verify %s\n' "$verify" >> release.intent
meta=$(sbatch --parsable --dependency=afterok:"$verify" --array=0-9%2 studies/16nov/followups/meta_adaptation/meta_array.sbatch)
printf 'meta %s\n' "$meta" >> release.intent
