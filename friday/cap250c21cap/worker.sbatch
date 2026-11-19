#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:14:00
#SBATCH --job-name=ni26-c21cap
#SBATCH --output=c21cap-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
native=$base/ni26_friday_c21e_native_v2_20261119
owner=$base/ni26_friday_c20e_native_v4_20261119
ref=$base/ni26_native360_sos_confirm_stress_v1_20261117
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
(cd "$native" && sha256sum --check --status source.sha256)
(cd "$owner" && sha256sum --check --status source.sha256)
(cd "$base" && sha256sum --check --status "$owner/parents.sha256")
(cd "$prior" && sha256sum --check --status inputs.sha256 && sha256sum --check --status members.sha256)
(cd "$ref" && echo '815138d6536adb999fdfdf37dfc243f66b3718b82bf8198ccd7cc04a1f1c53b5  reference_stress.json' | sha256sum --check --status)
grep -Fx OK "$prior/replay-proof.txt"
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/tmp,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/native:/v4:/prior \
    --bind "$work:/work:rw,$native:/native:ro,$owner:/v4:ro,$prior:/prior:ro,$prior/cap:/package:ro,$ref/reference_stress.json:/reference.json:ro,$repo/2026-competition:/official:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
    --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B probe.py
