#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s310-assemble
#SBATCH --output=logs/s310-assemble-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s310_entropy_sign_20261116/production_v2
source studies/16nov/followups/meta_adaptation/env.sh
export S310_S305=/mnt/lustre/work/gehler/gfh527/ni26_s305_converged_v2_20261116
export S310_S305_ASSEMBLER=/mnt/lustre/work/gehler/gfh527/ni26_s305_assembler_recover_v1_20261116
for f in source.sha256 pinned.sha256 cpu-src.sha256 s310-src.sha256; do sha256sum -c "$f" >/dev/null; done
"$PY" -B -I studies/16nov/followups/entropy_sign/assemble.py
chain=/mnt/lustre/work/gehler/gfh527/ni26_s310_entropy_sign_20261116/chain_v2
job=$(sbatch --parsable --chdir="$chain" --dependency=afterok:"$SLURM_JOB_ID" "$chain/studies/16nov/followups/entropy_sign/chain_dispatch.sbatch")
printf 'chain-dispatch %s\n' "$job" >> release.intent
