#!/bin/bash
# R19e family fits on Leonardo (array task = family 0..7, all its seeds sequentially, via worker.py).
# usage: _r19e_leo.sh TAG PEOPLE SMOKE(0|1) [FIRST_SESSION_ONLY]    (env ARR, T)
set -euo pipefail
W=/leonardo_work/AIFAC_P02_1636/ni26; R=$W/cap250/r19e_rel; tag=${1:?}; people=${2:?}; smoke=${3:?}; first=${4:-}
mkdir -p $R/logs $R/out/$tag
sbatch --parsable --account=${ACC:-AIFAC_P02_1636} -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t ${T:-01:00:00} \
  --array=${ARR:-0-7} -J r19e-$tag -o $R/logs/r19e-$tag-%A_%a.out --wrap "set -e; module load python/3.11.7; unset PYTHONPATH
source $W/.venv/bin/activate; cd $R
export CAP250_RELEASE=$R CAP250_ASSETS=$R CAP250_REV=${REV:?set REV=git rev-parse HEAD} CAP250_SMOKE=$smoke NI26_DEVICE=cuda PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export PYTHONPATH=$R/2026-competition/tracks/bci_decoding:$R/2026-competition
fam=(shallow_ea shallow_raw fbc_ea_fixed fbc_raw_fixed fbc_ea_fir fbc_raw_fir sinc_ea fbl_ea); n=(5 5 5 5 2 2 5 5)
f=\${fam[\$SLURM_ARRAY_TASK_ID]}
for ((s=0; s<\${n[\$SLURM_ARRAY_TASK_ID]}; s++)); do
  python -B cleanslate/cap250/worker.py --family \$f --seed \$s --people $people ${first:+--first-session-only $first} --out $R/out/$tag/\$f/s\$s
done"
