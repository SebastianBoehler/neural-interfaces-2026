#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=ni26-s308-cpu-smoke
#SBATCH --output=logs/cpu-smoke-%j.out
#SBATCH --error=logs/cpu-smoke-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export S308_WORKSPACE="$PWD"
source studies/16nov/followups/strong_heads/runtime.sh
export S308_SMOKE_OUT="$PWD/runs/smoke"
sha256sum -c source.sha256 >/dev/null
printf '%s  %s\n' 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 "$S308_BASE/data/proteus_windows.npz" | sha256sum -c -
printf '%s  %s\n' 6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f "$S308_BASE/runs/shuffle_perms.npz" | sha256sum -c -
grep -m1 'model name' /proc/cpuinfo
srun --cpu-bind=cores "$PYTHON" -B -I studies/16nov/followups/strong_heads/smoke_real.py
sha256sum -c source.sha256 >/dev/null
