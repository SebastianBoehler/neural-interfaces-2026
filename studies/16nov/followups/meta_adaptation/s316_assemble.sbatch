#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s316-assemble
#SBATCH --output=logs/s316-assemble-%j.out
set -euo pipefail
cd "${S316_TARGET_WS:?}"
source studies/16nov/followups/meta_adaptation/env.sh
for f in source.sha256 pinned.sha256 target-src.sha256; do sha256sum -c "$f" >/dev/null; done
"$PY" -B -I studies/16nov/followups/meta_adaptation/s316_assemble.py
job=$(sbatch --parsable --chdir="${S316_CHAIN_WS:?}" --dependency=afterok:"$SLURM_JOB_ID" "$S316_CHAIN_WS/studies/16nov/followups/meta_adaptation/chain_dispatch.sbatch")
printf 'chain-dispatch %s\n' "$job" >> release.intent
