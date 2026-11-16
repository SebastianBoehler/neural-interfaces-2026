#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s308-dispatch
#SBATCH --output=logs/dispatch-%j.out
#SBATCH --error=logs/dispatch-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source studies/16nov/followups/strong_heads/runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sha256sum -c source.sha256 >/dev/null
(cd "$S308_SMOKE_ROOT" && sha256sum -c source.sha256 >/dev/null)
(cd "$S308_CHAIN" && sha256sum -c s295-source.sha256 >/dev/null && sha256sum -c chain.sha256 >/dev/null)
test "$(sha256sum source.sha256 | cut -d' ' -f1)" = "$S308_PRODUCER_SHA"
test "$(sha256sum "$S308_SMOKE_ROOT/source.sha256" | cut -d' ' -f1)" = "$S308_SMOKE_SHA"
test "$(sha256sum "$S308_CHAIN/chain.sha256" | cut -d' ' -f1)" = "$S308_CHAIN_SHA"
srun --cpu-bind=cores "$PYTHON" -B -I studies/16nov/followups/strong_heads/guard.py
n=$(squeue -h -u "$USER" -t PD,R,CG -r -o '%A_%a' | wc -l)
if (( n + 23 > 272 )); then echo "Insufficient reserved submission capacity: active=$n need=23" >&2; exit 1; fi
jobs="$S308_WORKSPACE/runs/s308/jobs.txt"
test ! -e "$jobs"
export S308_WORKSPACE S308_CHAIN S308_PRODUCER_SHA S308_SMOKE_ROOT S308_SMOKE_SHA S308_CHAIN_SHA
fit=$(sbatch --parsable --array=0-19%2 studies/16nov/followups/strong_heads/produce.sbatch)
printf 'fit=%s\n' "$fit" | tee -a "$jobs"
assemble=$(sbatch --parsable --dependency="afterok:$fit" studies/16nov/followups/strong_heads/assemble.sbatch)
printf 'assemble=%s\n' "$assemble" | tee -a "$jobs"
stage=$(sbatch --parsable --chdir="$S308_CHAIN" --dependency="afterok:$assemble" "$S308_CHAIN/studies/16nov/followups/strong_heads/stage.sbatch")
printf 'stage=%s\n' "$stage" | tee -a "$jobs"
launcher=$(sbatch --parsable --chdir="$S308_CHAIN" --dependency="afterok:$stage" "$S308_CHAIN/studies/16nov/followups/strong_heads/chain_dispatch.sbatch")
printf 'chain_dispatch=%s\n' "$launcher" | tee -a "$jobs"
