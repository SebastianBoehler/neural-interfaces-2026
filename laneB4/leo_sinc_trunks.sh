#!/bin/bash
# Lane B4: extra Sinc trunks (m7r recipe, seeds 6-11) on Leonardo, then the Sinc LOPO fits for the same seeds (dependency afterok).
# Run from the release root: leo_sinc_trunks.sh      (pretrain.py recipe as scripts/audit_nxif.sh: 30 epochs, rotation 0.05, seven MI corpora)
set -euo pipefail
acc=AIFAC_P02_1675; root=$PWD
common="set -e; module load python/3.11.7; unset PYTHONPATH; source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1 BENCHOPT_DATA_HOME=$root/data LANEH_WORK=$root/work; cd $root"
mkdir -p logs work
P=$(sbatch --parsable --account=$acc -p boost_usr_prod --gres=gpu:1 -c 8 --mem=60G -t 03:00:00 --array=6-11 -J ni26-b4-sinctrunk -o logs/b4-trunk-%A_%a.out \
  --wrap "$common; python -u experiments/pretrain.py sincshallow ea m7r physionetmi,cho2017,lee2019_mi,dreyer2023,schirrmeister2017,shin2017a,shin2017b --epochs 30 --rot 0.05 --seed \$SLURM_ARRAY_TASK_ID")
F=$(sbatch --parsable --dependency=aftercorr:$P --account=$acc -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 01:00:00 --array=6-11 -J ni26-b4-sincfit -o logs/b4-sincfit-%A_%a.out \
  --wrap "$common; python -u laneB4/lopo_b4.py sinc \$SLURM_ARRAY_TASK_ID")
echo trunks $P fits $F
