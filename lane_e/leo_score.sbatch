#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-laneE-leo
#SBATCH --output=logs/leo-score-%x-%A_%a.out
# Lane E on Leonardo (A100, repo .venv; the worker image cannot use Leonardo's GPU driver). NI-V B-new task of one package:
#   sbatch --account=ACC --array=0-35 leo_score.sbatch ZIP TAG       task = person*4 + chunk, as niv/score.sbatch
# Arms of a task: full + within shuffles, order seeds 2c, 2c+1 (NI-V orders 71000+). Output out/TAG/p<person>_c<chunk>.npz.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset PYTHONPATH; module load python/3.11.7 >/dev/null 2>&1 || true
D=$PWD; PY=${LEO_REPO:-/leonardo_work/AIFAC_P02_1636/ni26}/.venv/bin/python
zipn=${1:?}; tag=${2:?}; t=${SLURM_ARRAY_TASK_ID:-0}; person=$((t/4)); chunk=$((t%4))
(cd "$D" && sha256sum --quiet -c zips.sha256 --ignore-missing)
arms="full:$((2*chunk)),full:$((2*chunk+1)),within:$((2*chunk)),within:$((2*chunk+1))"
mkdir -p out/$tag; out=out/$tag/p${person}_c${chunk}.npz; [ -e "$out" ] && { echo "exists $out"; exit 1; }
export NIV_OFFICIAL=${LEO_REPO:-/leonardo_work/AIFAC_P02_1636/ni26}/2026-competition NIV_EXTERNAL=$D/data/windows.npz NIV_ROWS=$D/data/scherer_rows.tsv.gz
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; hostname
$PY -I -B niv_leo/worker.py "zips/$zipn" "$out" "$person" "$arms"
