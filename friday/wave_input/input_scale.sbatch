#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 5
#SBATCH -J ni26-wave-scale
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --quiet -c actor.sha256
if [[ $1 == smoke ]]; then
    for kind in raw_scale_genuine raw_scale_donor; do
        ../venv/bin/python input_scale_fit.py "$kind" 460 P01 --smoke
    done
    ../venv/bin/python - <<'PY'
import json
from pathlib import Path
true, control = [json.loads(Path(f"smoke_raw_scale_{kind}_460_P01.json").read_text())
                 for kind in ("genuine", "donor")]
for key in ("reference_sha256", "donor_power_sha256", "train_rows", "initial_model",
            "episode_rows", "donor_episode_rows"):
    assert true[key] == control[key], key
assert true["checkpoint"] != control["checkpoint"]
print("F11 paired source rows, donor schedule, reference and initialization matched", flush=True)
PY
else
    seeds=(460 1460)
    task=${SLURM_ARRAY_TASK_ID:?}
    seed=${seeds[task/10]}
    printf -v held 'P%02d' "$(( task % 10 + 1 ))"
    ../venv/bin/python input_scale_fit.py "$1" "$seed" "$held"
fi
