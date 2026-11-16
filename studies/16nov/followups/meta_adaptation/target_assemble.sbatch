#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s307-assemble
#SBATCH --output=logs/s307-assemble-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/target_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
for f in source.sha256 pinned.sha256 target-src.sha256; do sha256sum -c "$f" >/dev/null; done
"$PY" -B -I studies/16nov/followups/meta_adaptation/assemble_target.py
chain=/mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/chain_v2
job=$(sbatch --parsable --chdir="$chain" --dependency=afterok:"$SLURM_JOB_ID" "$chain/studies/16nov/followups/meta_adaptation/chain_dispatch.sbatch")
printf 'chain-dispatch %s\n' "$job" >> target-release.intent
