#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s308-chain-dispatch
#SBATCH --output=logs/s308-chain-dispatch-%j.out
#SBATCH --error=logs/s308-chain-dispatch-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
test "$(sha256sum chain.sha256 | cut -d' ' -f1)" = "$S308_CHAIN_SHA"
sha256sum -c chain.sha256 >/dev/null
test -f runs/s301_native/manifest.json
n=$(squeue -h -u "$USER" -t PD,R,CG -r -o '%A_%a' | wc -l)
if (( n + 89 > 272 )); then echo "Insufficient reserved submission capacity: active=$n need=89" >&2; exit 1; fi
jobs="$S308_WORKSPACE/runs/s308/chain_jobs.txt"
test ! -e "$jobs"
zero=$(sbatch --parsable --array=0-7%2 studies/16nov/followups/strong_heads/member.sbatch zero)
printf 'zero=%s\n' "$zero" | tee -a "$jobs"
treatment=$(sbatch --parsable --array=0-79%2 --dependency="afterok:$zero" studies/16nov/followups/strong_heads/member.sbatch treatment)
printf 'treatment=%s\n' "$treatment" | tee -a "$jobs"
score=$(sbatch --parsable --dependency="afterok:$treatment" studies/16nov/followups/strong_heads/score.sbatch)
printf 'score=%s\n' "$score" | tee -a "$jobs"
