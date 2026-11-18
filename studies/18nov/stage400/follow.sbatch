#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,v100-galvani
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:10:00
#SBATCH -J ni26-st400b
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
arms=(L D)
if [[ $1 == score ]]; then exec "$py" follow_score.py; fi
if [[ $1 == smoke ]]; then
  "$py" cross.py RB within0
  "$py" cross.py L within0
  "$py" endpoint.py crossRB within0 B2_pool
  "$py" -c 'import numpy as n; from capture import ORIG; a=n.load("final_crossRB_within0_B2_pool.npz")["scores"]; b=n.load(ORIG/"runs/s281_v66_cp/RB_within0_B2_pool.npz")["scores"]; assert n.array_equal(a.argmax(1),b.argmax(1)) and n.allclose(a,b,rtol=0,atol=2e-6)'
else
  id=${SLURM_ARRAY_TASK_ID:?}
  arm=${arms[id/8]}
  order=${orders[id%8]}
  if [[ $id != 0 ]]; then "$py" cross.py "$arm" "$order"; fi
  for b in B2_pool B2_s345 B2_s012 B2_s0 B2_s1 B2_s2; do
    "$py" endpoint.py "cross$arm" "$order" "$b"
  done
fi
