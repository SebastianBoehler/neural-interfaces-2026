#!/bin/bash
# Lane T2 sweep on Leonardo (account AIFAC_P02_1675, boost_usr_prod). Run from the release root; login node only does sbatch.
# usage: sweep.sh run STAGE CFGJSON PEOPLE(comma) ORDERS(comma) NCHUNKS [TIME]
#   array task t: person = PEOPLE[t % nP], chunk = t / nP of NCHUNKS (configs sorted by name, every nth) -> work/STAGE/task_<P>_<chunk>.npz
set -euo pipefail
root=$PWD; mode=${1:?}
acc=AIFAC_P02_1675
stage=${2:?}; cfg=${3:?}; people=${4:?}; orders=${5:?}; nch=${6:?}; tl=${7:-01:00:00}
np_=$(echo $people | tr ',' '\n' | wc -l)
mkdir -p logs work/$stage
sbatch --parsable --account=$acc -p boost_usr_prod --gres=gpu:1 -c 4 --mem=24G -t $tl --array=0-$((np_*nch-1))${MAXPAR:+%$MAXPAR} \
  -J t2-$stage -o logs/$stage-%A_%a.out --wrap "set -e; module load python/3.11.7; unset PYTHONPATH; \
source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate; export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1 \
CS_DATA=$root/data/proteus_windows.npz CS_OFFICIAL=/leonardo_work/AIFAC_P02_1636/ni26/2026-competition/tracks/bci_decoding; cd $root; \
PL=($(echo $people | tr ',' ' ')); P=\${PL[\$((SLURM_ARRAY_TASK_ID % $np_))]}; C=\$((SLURM_ARRAY_TASK_ID / $np_)); \
python -u cleanslate/ttt_sweep/run_sweep.py --pkg pkgs/pkg_\$P --held \$P --orders $orders --cfgs $cfg --chunk \$C/$nch --out work/$stage/task_\${P}_\$C.npz"
