#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 5
#SBATCH -J ni26-wave-band
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --quiet -c actor.sha256
if [[ $1 == smoke ]]; then
    for kind in raw_band_spectral raw_band_random; do
        ../venv/bin/python fit.py "$kind" 460 P01 --smoke
    done
    ../venv/bin/python - <<'PY'
import json

a, b = (json.load(open(f"smoke_raw_band_{kind}_460_P01.json")) for kind in ("spectral", "random"))
for key in ("initial_other", "parameters", "train_rows", "episode_rows"):
    assert a[key] == b[key], key
assert all(abs(x-y) <= 1e-6 for x,y in zip(a["fir_norm"], b["fir_norm"], strict=True))
assert a["initial_fir"] != b["initial_fir"]
print("F12 matched non-FIR state, parameter count, episode rows and FIR norms")
PY
else
    seeds=(460 1460)
    task=${SLURM_ARRAY_TASK_ID:?}
    seed=${seeds[task/10]}
    printf -v held 'P%02d' "$((task % 10 + 1))"
    ../venv/bin/python fit.py "$1" "$seed" "$held"
fi
