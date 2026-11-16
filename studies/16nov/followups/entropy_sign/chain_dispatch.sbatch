#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s310-chain-stage
#SBATCH --output=logs/s310-chain-stage-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s310_entropy_sign_20261116/chain_v2
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
for f in s295-source.sha256 chain.sha256 s310-chain.sha256; do sha256sum -c "$f" >/dev/null; done
export S301_PRODUCER_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s310_entropy_sign_20261116/production_v2
export S301_PRODUCER_SHA="$(sha256sum "$S301_PRODUCER_ROOT/s310-src.sha256" | cut -d' ' -f1)"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
active=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
if (( active + 35 > 272 )); then
 retry=$(sbatch --parsable --begin=now+10minutes studies/16nov/followups/entropy_sign/chain_dispatch.sbatch)
 printf 'capacity %s; retry %s\n' "$active" "$retry" >> s310-chain-retries.log
 exit 0
fi
"$PY" -B -I studies/16nov/followups/entropy_sign/stage.py
set -o noclobber
printf 'stage %s\n' "$SLURM_JOB_ID" > s310-chain.intent
set +o noclobber
zero=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=0-7%2 studies/16nov/followups/entropy_sign/chain_job.sbatch zero)
printf 'zero %s\n' "$zero" >> s310-chain.intent
smoke=$(sbatch --parsable --dependency=afterok:"$zero" studies/16nov/followups/entropy_sign/chain_job.sbatch smoke)
printf 'smoke %s\n' "$smoke" >> s310-chain.intent
wave=$(sbatch --parsable --dependency=afterok:"$smoke" --array=1-23%2 studies/16nov/followups/entropy_sign/chain_job.sbatch treatment)
printf 'treatment %s\n' "$wave" >> s310-chain.intent
score=$(sbatch --parsable --dependency=afterok:"$wave" studies/16nov/followups/entropy_sign/chain_score.sbatch)
printf 'score %s\n' "$score" >> s310-chain.intent
