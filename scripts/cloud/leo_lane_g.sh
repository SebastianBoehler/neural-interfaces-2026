#!/bin/bash
# Submit Lane G (stronger personal model) to Leonardo. Usage (from anywhere):
#   leo_lane_g.sh ACCOUNT /abs/release/dir /abs/cap250.zip /abs/ni26_root
# release holds laneG/*.py (rsync'd); ni26_root holds data/proteus_windows.npz and runs/shuffle_perms.npz.
set -euo pipefail
account=${1:?account}; release=${2:?release}; zip=${3:?cap250 zip}; root=${4:?ni26 root}
[[ "$release" = /* && "$zip" = /* && "$root" = /* ]]
test -r "$zip"; test -r "$root/data/proteus_windows.npz"; test -r "$root/runs/shuffle_perms.npz"
mkdir -p "$release"/logs; code=$release/laneG
common=(--parsable --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=60G)
runtime="set -e; module load python/3.11.7; source '$root/.venv/bin/activate'; export NI26_ROOT='$root' PYTHONPATH='$code' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS=ignore; cd '$release'"
run="python '$code/jobs.py' '$zip' '$release/out'"
submit() { sbatch "${common[@]}" "$@" | cut -d ';' -f1; }
smoke=$(submit -J ni26-lg-smoke -t 00:30:00 -o "$release/logs/smoke-%j.out" \
  --wrap "$runtime; python '$code/tasks.py' '$zip' '$release/out' parity; $run feats 0 && test -s '$release/out/feats/P04_2_full0.npz' && echo SMOKE_OK")
feats=$(submit -J ni26-lg-feats -t 00:40:00 --array=0-3 -o "$release/logs/feats-%A_%a.out" --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
  --wrap "$runtime; $run feats \$SLURM_ARRAY_TASK_ID")
train=$(submit -J ni26-lg-train -t 00:50:00 --array=0-15 -o "$release/logs/train-%A_%a.out" --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
  --wrap "$runtime; $run train \$SLURM_ARRAY_TASK_ID")
prod=$(submit -J ni26-lg-prod -t 00:50:00 --array=0-9 -o "$release/logs/prod-%A_%a.out" --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
  --wrap "$runtime; $run prod \$SLURM_ARRAY_TASK_ID")
score=$(submit -J ni26-lg-score -t 00:20:00 -o "$release/logs/score-%j.out" --dependency="afterok:$feats:$train" --kill-on-invalid-dep=yes \
  --wrap "$runtime; python '$code/score.py' '$release/out' stock,g1,g1c,g1a --out '$release/out/score.json'")
printf 'smoke=%s feats=%s train=%s prod=%s score=%s\n' "$smoke" "$feats" "$train" "$prod" "$score" | tee "$release/job_ids.txt"
