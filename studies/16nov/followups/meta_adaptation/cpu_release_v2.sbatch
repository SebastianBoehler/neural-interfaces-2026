#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s307-cpu-release
#SBATCH --output=logs/s307-cpu-release-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
for f in source.sha256 pinned.sha256 cpu-src.sha256 dispatch.sha256; do sha256sum -c "$f" >/dev/null; done
cmp source.sha256 ../cpu_smoke_v2/source.sha256
cmp cpu-src.sha256 ../cpu_smoke_v2/cpu-src.sha256
cmp cache.sha256 ../cpu_smoke_v2/cache.sha256
test "$(sacct -X -n -P -j 3115208 -o State,ExitCode | head -1)" = 'COMPLETED|0:0'
sacct -X -n -P -j 3115208 -o JobID,Elapsed,State,ExitCode
"$PY" -B -I studies/16nov/followups/meta_adaptation/verify_cpu_pilot_v2.py
active=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
if (( active + 102 > 272 )); then
 retry=$(sbatch --parsable --begin=now+10minutes studies/16nov/followups/meta_adaptation/cpu_release_v2.sbatch)
 printf 'capacity %s; retry %s\n' "$active" "$retry" >> cpu-release-retries.log
 exit 0
fi
set -o noclobber
printf 'release %s\n' "$SLURM_JOB_ID" > cpu-release.intent
set +o noclobber
crossfit=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=0-89%4 studies/16nov/followups/meta_adaptation/cpu_crossfit_array_v2.sbatch)
printf 'crossfit %s\n' "$crossfit" >> cpu-release.intent
verify=$(sbatch --parsable --dependency=afterok:"$crossfit" studies/16nov/followups/meta_adaptation/verify_cpu_crossfit_v2.sbatch)
printf 'verify %s\n' "$verify" >> cpu-release.intent
gate=$(cat cpu-meta-gate.id)
if [[ -n $(squeue -h -j "$gate" -o %i) ]]; then dependency="$verify:$gate";
else
 test "$(sacct -X -n -P -j "$gate" -o State,ExitCode | head -1)" = 'COMPLETED|0:0'
 test -s cpu-meta-gate.ok
 dependency="$verify"
fi
meta=$(sbatch --parsable --dependency=afterok:"$dependency" --array=0-9%2 studies/16nov/followups/meta_adaptation/meta_cpu_v2.sbatch)
printf 'meta %s\n' "$meta" >> cpu-release.intent
