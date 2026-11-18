#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani,v100-galvani
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:10:00
#SBATCH -J ni26-r403
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
run() {
  "$py" transport.py "$1" "$2"
  "$py" endpoint.py "t$1" "$2" B2_pool
}
identity() {
  run RB "$1"
  "$py" -c 'import sys,numpy as n; from capture import ORIG; o=sys.argv[1]; a=n.load(f"final_tRB_{o}_B2_pool.npz")["scores"]; b=n.load(ORIG/f"runs/s281_v66_cp/control_{o}_B2_pool.npz")["scores"]; assert n.array_equal(a.argmax(1),b.argmax(1)) and n.allclose(a,b,rtol=0,atol=2e-6)' "$1"
}
case "$1" in
smoke) identity within0; run L within0; run D within0 ;;
score) exec "$py" report.py ;;
cell)
  n=${SLURM_ARRAY_TASK_ID:?}
  o=${orders[n%8]}
  if [[ "$o" != within0 ]]; then
    if ((n<8)); then identity "$o"; run L "$o"; else run D "$o"; fi
  fi ;;
esac
