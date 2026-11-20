#!/bin/bash
# Submit eye/rest/cbramod fits on Leonardo through worker.py (one array task per family x seed: id = 3*family + seed).
# usage: REV=<git revision> R=<release dir> _aux_slurm.sh TAG PEOPLE SMOKE(0|1) [FIRST_SESSION_ONLY]   (ARR=0-8 selects tasks)
set -euo pipefail
R=${R:?release dir}; REV=${REV:?git revision}; tag=${1:?}; people=${2:?}; smoke=${3:?}; first=${4:-}
W=/leonardo_work/AIFAC_P02_1636
blob=$W/ni26_fm280/hf/hub/models--braindecode--cbramod-pretrained/snapshots/584cdc415913739a05d84bf0c1cb3db397764507/model.safetensors
mkdir -p $R/logs $R/out/$tag
sbatch --parsable --account=AIFAC_P02_1636 -p boost_usr_prod --gres=gpu:1 -c 8 --mem=40G -t 01:00:00 \
  --array=${ARR:-0-8} -J aux-$tag -o $R/logs/aux-$tag-%A_%a.out --wrap "set -e; cd $R
module load python/3.11.7; unset PYTHONPATH; source $W/ni26/.venv/bin/activate
export CAP250_REV=$REV CAP250_RELEASE=$R CAP250_ASSETS=$R CAP250_SMOKE=$smoke CAP250_CB_CHECKPOINT=$blob NI26_DEVICE=cuda HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
export PYTHONPATH=$R/2026-competition/tracks/bci_decoding:$R/2026-competition
fam=(eye rest cbramod); f=\${fam[\$((SLURM_ARRAY_TASK_ID / 3))]}; s=\$((SLURM_ARRAY_TASK_ID % 3))
python -B cleanslate/cap250/worker.py --family \$f --seed \$s --people $people ${first:+--first-session-only $first} --out $R/out/$tag/\$f/s\$s"
