#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani,a100-preemptable-galvani,v100-galvani
#SBATCH --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-s312-fullprobe
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c schedule-source.sha256
export NI26_WS=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2 NI26_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2
export NI26_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_data1/cache
export NI26_OUT="$PWD/gpu_probe_output" NI26_DEVICE=cuda
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
cd "$NI26_WS"
sha256sum -c source.sha256 >/dev/null
PY=.venv/bin/python
"$PY" -c 'import torch;print("ACTUAL_GPU",torch.cuda.get_device_name(0),flush=True)'
/usr/bin/time -f 'COMPLETE_PRODUCER_SECONDS=%e MAXRSS_KB=%M' srun --cpu-bind=cores "$PY" run.py 80
