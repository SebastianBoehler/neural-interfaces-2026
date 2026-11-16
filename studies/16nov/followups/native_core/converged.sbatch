#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=ni26-s305-converged
#SBATCH --output=logs/converged-%A_%a.out
#SBATCH --error=logs/converged-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum converged.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export NI26_ROOT="$PWD"
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S301_PARITY_REPORT=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_parity_v3_20261116/runs/s301_parity/report.json
export S301_ALL_REFERENCES=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
unset PYTHONPATH
sha256sum -c source.sha256 >/dev/null
sha256sum -c producer.sha256 >/dev/null
sha256sum -c converged.sha256
printf '%s  %s\n' 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 data/proteus_windows.npz | sha256sum -c -
printf '%s  %s\n' 6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f runs/shuffle_perms.npz | sha256sum -c -
printf '%s  %s\n' ce5b045357675aaa36e6b9b6722d357c4062fcba5340e23f09fdc3d9a353d479 runs/coremember.npz | sha256sum -c -
printf '%s  %s\n' 4aad0f4d19c05092c6eb8748798f2146fd05891cc3c2b78ae85a0d0ccec77425 runs/coremember_within2_within3_full1_runs1.npz | sha256sum -c -
(cd /mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116 && sha256sum -c runs/s301_full_build/library.sha256)
(cd /mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116 && sha256sum -c runs/s301_trainref/library.sha256)
(cd /mnt/lustre/work/gehler/gfh527/ni26_s301_native_audit_20261116 && sha256sum -c audit.sha256 >/dev/null)
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/converged_gate.py
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/converged.py "${SLURM_ARRAY_TASK_ID:?}"
