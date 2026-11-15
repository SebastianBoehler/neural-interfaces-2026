#!/bin/bash
# s280 REVE on Leonardo (run from $WORK/ni26_reve280 on the login node; only sbatch). Usage: reve_leo.sh smoke | array
set -euo pipefail
ACC=${ACC:-AIFAC_P02_1636}; mkdir -p logs
PRE='module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate; export NI26_WS=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore; python -c "import torch; assert torch.cuda.is_available()" || exit 1'
case $1 in
smoke) sbatch --parsable --account=$ACC -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=00:30:00 -J reve280-smoke -o logs/smoke-%j.out --wrap "$PRE
for f in rv1 rv2 rv3 rv0; do python studies/15nov/fm280/reve_fm.py smoke \$f || exit 1; done
python studies/15nov/fm280/reve_cost.py cuda; echo EXIT \$?";;
array) sbatch --parsable --account=$ACC -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=60G --time=${TIME:-04:00:00} ${DEP:-} --array=0-23 -J reve280 -o logs/reve280-%A_%a.out --wrap "$PRE
python studies/15nov/fm280/reve_fm.py \$SLURM_ARRAY_TASK_ID; rc=\$?; echo EXIT \$rc; exit \$rc";;
esac
