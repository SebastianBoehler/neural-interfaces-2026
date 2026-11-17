#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s316-release
#SBATCH --output=logs/s316-release-%j.out
set -euo pipefail
cd "${S316_TARGET_WS:?}"
for f in source.sha256 pinned.sha256 target-src.sha256 meta-cpu-src.sha256 meta.sha256; do sha256sum -c "$f" >/dev/null; done
pilot="${S316_PILOT_WS:?}"
pilot_sha=$(sha256sum "$pilot/runs/s307/target/P09.npz" | cut -d' ' -f1)
source_sha=$(sha256sum target-src.sha256 | cut -d' ' -f1)
test "$(sha256sum "$pilot/target-src.sha256" | cut -d' ' -f1)" = "$source_sha"
test "$(cat "$pilot/runs/s307/target/P09.cpu-route")" = "CPU $pilot_sha $source_sha"
active=$(squeue -h -r -u "$USER" -t PENDING,RUNNING,COMPLETING -o '%i' | wc -l)
running=$(squeue -h -r -u "$USER" -t RUNNING -o '%i' | wc -l)
if (( active + 12 > 272 || running + 2 > 32 )); then
  sbatch --parsable --begin=now+10minutes studies/16nov/followups/meta_adaptation/s316_release.sbatch
  exit 0
fi
test ! -e release.intent
test ! -e runs/s307/target/P01.npz
printf 'release %s\n' "$SLURM_JOB_ID" > release.intent
array=$(sbatch --parsable --array=0-9%2 studies/16nov/followups/meta_adaptation/s316_cell.sbatch)
printf 'target %s\n' "$array" >> release.intent
assembly=$(sbatch --parsable --dependency=afterok:"$array" studies/16nov/followups/meta_adaptation/s316_assemble.sbatch)
printf 'assembly %s\n' "$assembly" >> release.intent
