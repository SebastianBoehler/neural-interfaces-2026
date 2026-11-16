#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s307-chain-stage
#SBATCH --output=logs/s307-chain-stage-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/chain_v2
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
for f in s295-source.sha256 chain.sha256 s307-chain.sha256; do sha256sum -c "$f" >/dev/null; done
export S301_PRODUCER_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/target_cpu_v2
export S301_PRODUCER_SHA="$(sha256sum "$S301_PRODUCER_ROOT/target-src.sha256" | cut -d' ' -f1)"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
active=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
if (( active + 75 > 272 )); then
 retry=$(sbatch --parsable --begin=now+10minutes studies/16nov/followups/meta_adaptation/chain_dispatch.sbatch)
 printf 'capacity %s; retry %s\n' "$active" "$retry" >> s307-chain-retries.log
 exit 0
fi
"$PY" -B -I studies/16nov/followups/meta_adaptation/chain_stage.py
set -o noclobber
printf 'stage %s\n' "$SLURM_JOB_ID" > s307-chain.intent
set +o noclobber
zero=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=0-7%2 studies/16nov/followups/meta_adaptation/chain_job.sbatch zero)
printf 'zero %s\n' "$zero" >> s307-chain.intent
smoke=$(sbatch --parsable --dependency=afterok:"$zero" studies/16nov/followups/meta_adaptation/chain_job.sbatch smoke)
printf 'smoke %s\n' "$smoke" >> s307-chain.intent
wave=$(sbatch --parsable --dependency=afterok:"$smoke" --array=1-63%2 studies/16nov/followups/meta_adaptation/chain_job.sbatch treatment)
printf 'treatment %s\n' "$wave" >> s307-chain.intent
score=$(sbatch --parsable --dependency=afterok:"$wave" studies/16nov/followups/meta_adaptation/chain_score.sbatch)
printf 'score %s\n' "$score" >> s307-chain.intent
