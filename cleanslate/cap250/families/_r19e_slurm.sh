#!/bin/bash
# Submit R19e family fits (array task = family 0..7, all its seeds sequentially) through worker.py on Galvani.
# usage: R=<release dir> _r19e_slurm.sh TAG PEOPLE SMOKE(0|1) [FIRST_SESSION_ONLY]
set -euo pipefail
R=${R:?release dir}; tag=${1:?}; people=${2:?}; smoke=${3:?}; first=${4:-}
M=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
mkdir -p $R/logs $R/out/$tag
sbatch --parsable --account=gehler -p a100-galvani,a100-preemptable-galvani --gres=gpu:1 -c 4 --mem=24G -t 01:00:00 \
  --array=${ARR:-0-7} -J r19e-$tag -o $R/logs/r19e-$tag-%A_%a.out --wrap "set -e; cd $R
export CAP250_RELEASE=$R CAP250_ASSETS=$R CAP250_SMOKE=$smoke NI26_DEVICE=cuda PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export PYTHONPATH=$R/2026-competition/tracks/bci_decoding:$R/2026-competition
fam=(shallow_ea shallow_raw fbc_ea_fixed fbc_raw_fixed fbc_ea_fir fbc_raw_fir sinc_ea fbl_ea); n=(5 5 5 5 2 2 5 5)
f=\${fam[\$SLURM_ARRAY_TASK_ID]}
for ((s=0; s<\${n[\$SLURM_ARRAY_TASK_ID]}; s++)); do
  $M/.venv/bin/python -B cleanslate/cap250/worker.py --family \$f --seed \$s --people $people ${first:+--first-session-only $first} --out $R/out/$tag/\$f/s\$s
done"
