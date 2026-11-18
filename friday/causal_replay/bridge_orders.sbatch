#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-causal-orders
#SBATCH --output=orders-%A_%a.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
RUN=$BASE/ni26_friday_causal_rebuild_20261118/orders_v2
cd "$RUN"
sha256sum --quiet -c source.sha256
(
    cd "$BASE/ni26_s318_coral_fbl_20261117"
    sha256sum --quiet -c "$RUN/owner.sha256"
    sha256sum --quiet -c "$RUN/recognition.sha256"
    sha256sum --quiet -c "$RUN/order_inputs.sha256"
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
orders=(within0 within1 within2 within3 full1 runs0 runs1)
order=${orders[${SLURM_ARRAY_TASK_ID:-0}]}
exec "$BASE/neural-interfaces-2026/.venv/bin/python" -B bridge_orders.py "$order" "$RUN/$order"
