#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-304-recovery-release
#SBATCH --output=logs/304-recovery-release-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s304_coral_chain_recovery_20261117
for f in chain304-src.sha256 stage-src.sha256 coral304-chain-src.sha256 recovery.sha256 runs/coral304_inputs/registration.sha256; do sha256sum -c "$f" >/dev/null; done
check=${1:?}; [[ "$(sacct -X -n -P -j "$check" -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]]
grep -qx OK "logs/304-recovery-check-$check.out"
grep -q 'OK: §304 zero/add/swap gate' "logs/304-recovery-check-$check.out"
active=$(squeue -h -u "$USER" -r -o '%i' | wc -l);running=$(squeue -h -u "$USER" -r -t RUNNING -o '%i' | wc -l)
if (( active + 70 > 272 || running + 4 > 32 )); then
 retry=$(sbatch --parsable --begin=now+10minutes studies/16nov/followups/coral_repair/chain/recovery/release.sbatch "$check")
 printf 'capacity %s/%s retry %s\n' "$active" "$running" "$retry" >> recovery-retries.log
 exit 0
fi
set -o noclobber
printf 'release %s\n' "$SLURM_JOB_ID" > runs/coral304_inputs/recovery.intent
set +o noclobber
wave=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=1-39,41-63,65-71%4 studies/16nov/followups/coral_repair/chain/recovery/job.sbatch)
printf 'wave %s\n' "$wave" >> runs/coral304_inputs/recovery.intent
score=$(sbatch --parsable --dependency=afterok:"$wave" studies/16nov/followups/coral_repair/chain/recovery/score.sbatch)
printf 'score %s\n' "$score" >> runs/coral304_inputs/recovery.intent
