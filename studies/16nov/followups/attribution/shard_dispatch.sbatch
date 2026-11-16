#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-300-queue
#SBATCH --output=logs/300-shard-queue-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s300_shards_20261116/system
sha256sum -c shard-src.sha256
sha256sum -c shard-v2-src.sha256
sha256sum -c collect-v2-src.sha256
sha256sum -c shard-dispatch.sha256
sha256sum -c attribution-registration.sha256
[[ "$(sacct -X -n -P -j 3114404 -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]]
grep -q 'OK: replacement smoke source hashes' logs/300-order-3114404_*.out
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/attribution/check_shard.py
oldlog=/mnt/lustre/work/gehler/gfh527/ni26_lanes294_20261116/system/logs/300-attribution-3114212.out
if grep -q '§300 exploratory saved-stage attribution' "$oldlog"; then
 echo 'serial attribution already emitted a guarded result; replacement not submitted'; exit 1
fi
oldstate=$(squeue -h -j 3114212 -o %T)
[[ "$oldstate" == RUNNING ]] || { echo "serial job is not running: $oldstate"; exit 1; }
running=$(squeue -h -u "$USER" -r -t RUNNING | wc -l)
(( running - 1 + 8 <= 32 )) || { echo "unsafe allocation capacity: $running"; exit 1; }
[[ ! -e runs/lane1_inputs/s300-shards.intent ]] || { echo 'shards already attempted'; exit 1; }
( set -o noclobber; printf 'dispatch %s after smoke 3114404\n' "$SLURM_JOB_ID" > runs/lane1_inputs/s300-shards.intent )
scancel 3114212
printf 'cancelled serial 3114212 after smoke; no score emitted\n' | tee -a runs/lane1_inputs/s300-shards.intent
wave=$(sbatch --parsable --array=0-7%8 studies/16nov/followups/attribution/shard_v2.sbatch)
printf 'wave %s\n' "$wave" | tee -a runs/lane1_inputs/s300-shards.intent
score=$(sbatch --parsable --dependency="afterok:$wave" studies/16nov/followups/attribution/collect_v2.sbatch)
printf 'score %s\n' "$score" | tee -a runs/lane1_inputs/s300-shards.intent
