#!/usr/bin/env bash
# Leonardo: stream NEW orders through the existing fold packages of production-20261120-32a94ab7 (stock vs hardcb_safe).
# usage: pkgcmp_submit.sh smoke | wave "71004,71005,71006,71007" [more chunks...]   (sbatch only on the login node)
set -euo pipefail
W=/leonardo_work/AIFAC_P02_1636/ni26
REL=$W/cap250/releases/15a639718601e959015fd533fa9106476614aa8c
SRC=$W/cap250/runs/production-20261120-32a94ab7
RUN=$W/cap250/runs/overnight-21nov-pkgcmp
mkdir -p "$RUN/logs"
export CAP250_RELEASE=$REL CAP250_RUN=$RUN CAP250_ASSETS=$W/cap250/assets CAP250_DATA=$W/data/proteus_windows.npz \
  CAP250_PYTHON=$W/.venv312/bin/python CAP250_SITE=leonardo CAP250_REV=15a63971 CAP250_SMOKE=0 \
  CAP250_CB_STAGEA=/leonardo_work/AIFAC_P02_1636/ni26_aux_20261120/runs/cb_stage_a \
  CAP250_CB_CHECKPOINT=$W/cap250/assets/runs/cb_stage_a/cbramod.safetensors
SB=(sbatch --parsable --account=AIFAC_P02_1636 --partition=boost_usr_prod --gres=gpu:1 --chdir=$RUN)
mode=${1:?}; shift
if [[ ! -f $RUN/prep.jobid ]]; then
  PREP=$("${SB[@]}" --job-name=ov21-prep --cpus-per-task=2 --mem=4G --time=20 --output=$RUN/logs/%x-%j.out \
    --wrap="python3 $W/cap250/overnight21/pkgcmp_prep.py $SRC $RUN")
  echo "$PREP" > $RUN/prep.jobid; echo "prep $PREP"
fi
PREP=$(cat $RUN/prep.jobid)
# prep is a 5 s job; once Slurm has purged it afterok is rejected, so depend on it only while it is still queued/running
if [[ -z $(squeue -h -j $PREP 2>/dev/null) ]]; then [[ $(sacct -j $PREP -X -n -o State | tr -d " ") == COMPLETED* ]] || { echo "prep not completed"; exit 1; }; PREP=none; fi
stream() { # name pkg held sessions orders out depflag minutes
  "${SB[@]}" --job-name=$1 --cpus-per-task=4 --mem=16G --time=$8 ${7/#none/} --output=$RUN/logs/%x-%j.out \
    --wrap="bash $REL/cleanslate/cap250/slurm.sh cleanslate/cap250/stream_fold.py --pkg $2 --held $3 --sessions $4 --orders $5 --data $CAP250_DATA --out $6"
}
if [[ $mode == smoke ]]; then
  J=$(stream ov21-smoke $RUN/hardcb_safe/new_P01/pkg P01 all 71000 $RUN/smoke_pred_new_P01.npz ${PREP/#[0-9]*/--dependency=afterok:$PREP} 20); echo "smoke stream $J"; exit 0
fi
for orders in "$@"; do
  tag=o${orders%%,*}; D=$RUN/$tag; mkdir -p $D
  for arm in ${ARMS:-stock hardcb_safe}; do
    mkdir -p $D/$arm; cp $SRC/hardcb/jobs.tsv $D/$arm/jobs.tsv; deps=""
    while IFS=$'\t' read -r kind held _ enrol ses; do
      sess=all; [[ $kind == ret ]] && sess=$ses
      J=$(stream ov21-$arm-$kind$held-$tag $RUN/$arm/${kind}_$held/pkg $held $sess $orders $D/$arm/pred_${kind}_$held.npz ${PREP/#[0-9]*/--dependency=afterok:$PREP} 45)
      deps+=":$J"
    done < $SRC/hardcb/jobs.tsv
    S=$("${SB[@]}" --job-name=ov21-score-$arm-$tag --cpus-per-task=2 --mem=8G --time=20 --dependency=afterok$deps --output=$RUN/logs/%x-%j.out \
      --wrap="bash $REL/cleanslate/cap250/slurm.sh cleanslate/harness/score.py $D/$arm --data $CAP250_DATA")
    echo "$arm $tag score $S"
  done
done
