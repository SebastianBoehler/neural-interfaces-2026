#!/usr/bin/env bash
# Lane FT-labels (Leonardo): stream orders through the FLIPPED fold packages (v6_ft_labels=1). Run on the login node (sbatch only).
# usage: ftl_submit.sh prep | smoke | wave "71000,71001,71002,71003" ["71004,..." ...]     ARMS="stock hardcb_safe" by default
set -euo pipefail
W=/leonardo_work/AIFAC_P02_1636/ni26
REL=$W/cap250/releases/15a639718601e959015fd533fa9106476614aa8c
SRC=$W/cap250/runs/production-20261120-32a94ab7
PKG=$W/cap250/runs/overnight-21nov-pkgcmp
RUN=$W/cap250/runs/laneFT-labels
mkdir -p "$RUN/logs"
export CAP250_RELEASE=$REL CAP250_RUN=$RUN CAP250_ASSETS=$W/cap250/assets CAP250_DATA=$W/data/proteus_windows.npz \
  CAP250_PYTHON=$W/.venv312/bin/python CAP250_SITE=leonardo CAP250_REV=15a63971 CAP250_SMOKE=0 \
  CAP250_CB_STAGEA=/leonardo_work/AIFAC_P02_1636/ni26_aux_20261120/runs/cb_stage_a \
  CAP250_CB_CHECKPOINT=$W/cap250/assets/runs/cb_stage_a/cbramod.safetensors
SB=(sbatch --parsable --account=AIFAC_P02_1636 --partition=boost_usr_prod --gres=gpu:1 --chdir=$RUN)
mode=${1:?}; shift
stream() { # name pkg held sessions orders out minutes [dependency]
  "${SB[@]}" --job-name=$1 --cpus-per-task=4 --mem=16G --time=$7 ${8:-} --output=$RUN/logs/%x-%j.out \
    --wrap="bash $REL/cleanslate/cap250/slurm.sh cleanslate/cap250/stream_fold.py --pkg $2 --held $3 --sessions $4 --orders $5 --data $CAP250_DATA --out $6"
}
if [[ $mode == prep ]]; then
  "${SB[@]}" --job-name=ftl-prep --cpus-per-task=2 --mem=16G --time=30 --output=$RUN/logs/%x-%j.out \
    --wrap="rm -rf $RUN/pkgs && $W/.venv312/bin/python $W/cap250/overnight21/ftl_prep.py $PKG $RUN/pkgs"; exit 0
fi
if [[ $mode == smoke ]]; then
  # H1 (flag + guard patch), H2 (teacher reads the hardcb mix), T0 (H2 code, flag 0: must equal stored hardcb)
  stream ftl-smoke-h1 $RUN/pkgs/hardcb_safe/new_P01/pkg P01 all 71000 $RUN/smoke_h1_new_P01.npz 20
  stream ftl-smoke-h2 $RUN/pkgs/hardcb_teach/new_P01/pkg P01 all 71000 $RUN/smoke_h2_new_P01.npz 20
  stream ftl-smoke-t0 $RUN/pkgs/smoke_t0/new_P01/pkg P01 all 71000 $RUN/smoke_t0_new_P01.npz 20
  stream ftl-smoke-s $RUN/pkgs/stock/new_P01/pkg P01 all 71000 $RUN/smoke_s_new_P01.npz 20; exit 0
fi
for orders in "$@"; do
  tag=o${orders%%,*}; D=$RUN/$tag; mkdir -p $D
  for arm in ${ARMS:-stock hardcb_safe}; do
    mkdir -p $D/$arm; cp $SRC/hardcb/jobs.tsv $D/$arm/jobs.tsv; deps=""
    while IFS=$'\t' read -r kind held _ enrol ses; do
      sess=all; [[ $kind == ret ]] && sess=$ses
      J=$(stream ftl-$arm-$kind$held-$tag $RUN/pkgs/$arm/${kind}_$held/pkg $held $sess $orders $D/$arm/pred_${kind}_$held.npz 45)
      deps+=":$J"
    done < $SRC/hardcb/jobs.tsv
    S=$("${SB[@]}" --job-name=ftl-score-$arm-$tag --cpus-per-task=2 --mem=8G --time=20 --dependency=afterok$deps --output=$RUN/logs/%x-%j.out \
      --wrap="bash $REL/cleanslate/cap250/slurm.sh cleanslate/harness/score.py $D/$arm --data $CAP250_DATA")
    echo "$arm $tag score $S"
  done
done
