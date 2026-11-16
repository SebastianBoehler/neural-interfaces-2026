#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani,2080-galvani,a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=ni26-s307-meta
#SBATCH --output=logs/s307-meta-%A_%a.out
#SBATCH --error=logs/s307-meta-%A_%a.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/production_cpu_v2
source studies/16nov/followups/meta_adaptation/env.sh
sha256sum -c source.sha256 >/dev/null
sha256sum -c pinned.sha256 >/dev/null
sha256sum -c dispatch.sha256 >/dev/null
sha256sum -c meta-cpu-src.sha256 >/dev/null
n=${SLURM_ARRAY_TASK_ID:?}; (( n >= 0 && n < 10 )); printf -v outer 'P%02d' "$((n+1))"
awk -v prefix="runs/s307/crossfit/${outer}_" '$2 ~ "^" prefix { print }' crossfit.sha256 > "logs/crossfit-${outer}.sha256"
test "$(wc -l < "logs/crossfit-${outer}.sha256")" = 9
sha256sum -c "logs/crossfit-${outer}.sha256" >/dev/null
for seed in 0 1; do
 for kind in scalar blocks; do
  "$PY" -B -I studies/16nov/followups/meta_adaptation/meta_train_cpu.py "$outer" "$kind" "$seed"
  path="runs/s307/meta/${outer}_${kind}_s${seed}.npz"
  set -o noclobber
  printf 'CPU %s %s\n' "$(sha256sum "$path" | cut -d' ' -f1)" "$(sha256sum meta-cpu-src.sha256 | cut -d' ' -f1)" > "${path%.npz}.cpu-route"
  set +o noclobber
 done
done
