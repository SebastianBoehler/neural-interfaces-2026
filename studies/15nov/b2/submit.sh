#!/bin/bash
# s284 B2 on Leonardo, in the work dir: prep (core1 features, s263 variant Z), then a 40-task array (task = W_index*10 + fold) afterok.
cd "$(dirname "$0")/../../.." || exit 1; mkdir -p runs/b2/logs
ENV='export PATH=$PWD/.venv/bin:$PATH OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS=ignore NI26_ROOT=$PWD; '
COM="-p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=60G ${ACC:+-A $ACC}"
P=$(sbatch --parsable $COM --time=01:00:00 -J b2prep -o runs/b2/logs/prep-%j.out --wrap "$ENV python -I studies/15nov/b2/b2.py prep")
F=$(sbatch --parsable $COM --time=02:00:00 --array=0-39 --requeue --dependency=afterok:$P -J b2fold -o runs/b2/logs/fold-%A_%a.out \
  --wrap "$ENV W=(0 0.1 0.3 1); python -I studies/15nov/b2/b2.py fold \${W[\$((SLURM_ARRAY_TASK_ID/10))]} \$((SLURM_ARRAY_TASK_ID%10))")
echo prep $P folds $F
