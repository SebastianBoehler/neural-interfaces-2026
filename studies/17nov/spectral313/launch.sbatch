#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s313-launch
#SBATCH --output=logs/launch-%j.out
#SBATCH --error=logs/launch-%j.err
set -euo pipefail
cd "${S313_CHAIN:?}"
test "$(sha256sum chain.sha256 | cut -d' ' -f1)" = "$S313_CHAIN_SHA"
sha256sum -c chain.sha256 >/dev/null
test -f runs/s301_native/manifest.json
n=$(squeue -h -u "$USER" -t PD,R,CG -r -o '%A_%a' | wc -l)
if (( n + 49 > 272 )); then echo "Insufficient reserved capacity: active=$n need=49" >&2; exit 1; fi
jobs="$S313_COLLECT_WS/runs/s313_collect/chain_jobs.txt"
test ! -e "$jobs"
zero=$(sbatch --parsable --array=0-7%2 studies/17nov/spectral313/member.sbatch zero)
printf 'zero=%s\n' "$zero" | tee -a "$jobs"
treatment=$(sbatch --parsable --array=0-39%2 --dependency="afterok:$zero" studies/17nov/spectral313/member.sbatch treatment)
printf 'treatment=%s\n' "$treatment" | tee -a "$jobs"
score=$(sbatch --parsable --dependency="afterok:$treatment" studies/17nov/spectral313/score_chain.sbatch)
printf 'score=%s\n' "$score" | tee -a "$jobs"
