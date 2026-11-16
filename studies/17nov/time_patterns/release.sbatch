#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s314-release
#SBATCH --output=logs/s314-release-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s314_time_patterns_20261117/production
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c s314-source.sha256 >/dev/null
cmp s314-source.sha256 ../smoke/s314-source.sha256
pilot=$(cat pilot.id)
test "$(sacct -X -n -P -j "$pilot" -o State,ExitCode | head -1)" = 'COMPLETED|0:0'
sacct -X -n -P -j "$pilot" -o JobID,Elapsed,State,ExitCode
(cd ../smoke && NI26_WS="$PWD" "$PY" -B -I studies/17nov/time_patterns/smoke_check.py)
active=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
if (( active + 22 > 272 )); then
 retry=$(sbatch --parsable --begin=now+10minutes studies/17nov/time_patterns/release.sbatch)
 printf 'capacity %s retry %s\n' "$active" "$retry" >> release-retries.log
 exit 0
fi
set -o noclobber
printf 'release %s\n' "$SLURM_JOB_ID" > release.intent
set +o noclobber
wave=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=0-19%2 studies/17nov/time_patterns/job.sbatch)
printf 'fit %s\n' "$wave" >> release.intent
collect=$(sbatch --parsable --dependency=afterok:"$wave" studies/17nov/time_patterns/collect.sbatch)
printf 'collect %s\n' "$collect" >> release.intent
