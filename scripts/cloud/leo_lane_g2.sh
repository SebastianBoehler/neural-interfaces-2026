#!/bin/bash
# Lane G amendment 1: arms g2, g4 (covariance-safe augmentation) + stock control, then the combined score with release 1's g1/g1c.
#   leo_lane_g2.sh ACCOUNT /abs/release2 /abs/cap250.zip /abs/ni26_root /abs/release1 DEP_RELEASE1_JOBS(colon list, e.g. 1:2)
set -euo pipefail
account=${1:?}; release=${2:?}; zip=${3:?}; root=${4:?}; r1=${5:?}; dep=${6:?}
mkdir -p "$release"/logs; code=$release/laneG
common=(--parsable --account="$account" -p boost_usr_prod --gres=gpu:1 -c 8 --mem=60G)
runtime="set -e; module load python/3.11.7; source '$root/.venv/bin/activate'; export NI26_ROOT='$root' PYTHONPATH='$code' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS=ignore; cd '$release'"
submit() { sbatch "${common[@]}" "$@" | cut -d ';' -f1; }
train=$(submit -J ni26-lg2-train -t 00:50:00 --array=0-11 -o "$release/logs/train-%A_%a.out" \
  --wrap "$runtime; python '$code/jobs.py' '$zip' '$release/out' train2 \$SLURM_ARRAY_TASK_ID")
score=$(submit -J ni26-lg2-score -t 00:20:00 -o "$release/logs/score-%j.out" --dependency="afterok:$train:$dep" --kill-on-invalid-dep=yes \
  --wrap "$runtime; mkdir -p out/feats out/arms out/pkg; cp -n '$r1'/out/feats/* out/feats/; cp -n '$r1'/out/arms/*_g1.npz '$r1'/out/arms/*_g1c.npz out/arms/; cp -n '$r1'/out/pkg/weights.npz out/pkg/ || true; python '$code/score.py' out stock,g1,g1c,g2,g4 --out out/score.json")
printf 'train2=%s score=%s\n' "$train" "$score" | tee "$release/job_ids.txt"
