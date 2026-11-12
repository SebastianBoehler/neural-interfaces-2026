#!/bin/bash
# Leonardo smoke test: the S3 and NETAUG smoke paths on one A100. Usage (from $WORK/ni26): scripts/cloud/leo_smoke.sh ACCOUNT
set -euo pipefail
mkdir -p runs/diag
sbatch --parsable --account=$1 -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00 -J ni26-smoke -o runs/diag/leo_smoke-%j.out --wrap "
module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate; export NI26_WS=\$PWD PYTHONWARNINGS=ignore
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
python -I studies/12nov/sys3.py smoke && python -I studies/12nov/netaug.py smoke_dro && echo LEO_SMOKE_OK"
