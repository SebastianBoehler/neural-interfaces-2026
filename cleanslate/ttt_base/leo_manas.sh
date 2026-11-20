#!/bin/bash
# MANAS fold packages (team A, b6-a1-manas new_Pxx, selected held-new fit) streamed under ttt_base orders; one array task per held person.
# usage: leo_manas.sh [ARR=0-9]  from the release root. Output work/manas/M_Pxx.npz
set -euo pipefail
root=$PWD; acc=AIFAC_P02_1636; A=/leonardo_work/AIFAC_P02_1675/ni26_cleanslate_A2/b6-a1-manas
mkdir -p logs work/manas
sbatch --parsable --account=$acc -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 02:00:00 --array=${ARR:-0-9} -J tb-manas -o logs/manas-%A_%a.out --wrap "set -e; module load python/3.11.7; unset PYTHONPATH; source /leonardo_work/AIFAC_P02_1636/ni26/.venv312/bin/activate; export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1; cd $root; P=\$(printf 'P%02d' \$((SLURM_ARRAY_TASK_ID+1))); python -u cleanslate/ttt_base/manas_stream.py --pkg $A/new_\$P --held \$P --orders 0,1 --out work/manas/M_\$P.npz"
