#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-dea-shallow
#SBATCH --output=shallow-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
ANCHOR=$BASE/ni26_friday_ea_anchors_20261118
cd "$BASE/ni26_friday_ea_shallow_20261118"
sha256sum --quiet -c run.sha256
(
    cd "$ANCHOR"
    sha256sum --quiet -c source.sha256
    sha256sum --quiet -c archive.sha256
)
revision=c625ecc736a881836a25f85a741ea9147bf698bf
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export BENCHOPT_DATA_HOME="$REPO/data"
export PYTHONPATH="$ANCHOR:$ANCHOR/$revision/experiments:$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
phase=${1:?expected inputs or reconstruct}
[[ "$phase" != reconstruct ]] || nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$REPO/.venv/bin/python" -B recording.py "$REPO" "$phase"
