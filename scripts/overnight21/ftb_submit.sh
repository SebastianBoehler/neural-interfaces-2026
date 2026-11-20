#!/usr/bin/env bash
# Lane FT-B6s (Leonardo, account AIFAC_P02_1636). Login node: sbatch only.
# usage: ftb_submit.sh smoke | pretrain | fit | pack | stream "71000,71001,71002,71003" ...
set -euo pipefail
W=/leonardo_work/AIFAC_P02_1636/ni26
B6=/leonardo_work/AIFAC_P02_1675/ni26_laneB6_20261120
REL=$W/cap250/releases/15a639718601e959015fd533fa9106476614aa8c
SRC=$W/cap250/runs/production-20261120-32a94ab7
PKG=$W/cap250/runs/overnight-21nov-pkgcmp
RUN=$W/cap250/runs/laneFT-b6s
ASSETS=$W/cap250/assets
OV=$W/cap250/overnight21
mkdir -p "$RUN/logs" "$RUN/parts"
export CAP250_RELEASE=$REL CAP250_RUN=$RUN CAP250_ASSETS=$ASSETS CAP250_DATA=$W/data/proteus_windows.npz \
  CAP250_PYTHON=$W/.venv312/bin/python CAP250_SITE=leonardo CAP250_REV=15a63971 CAP250_SMOKE=${SMOKE:-0} FTB_JOBS=$SRC/hardcb/jobs.tsv
SB=(sbatch --parsable --account=AIFAC_P02_1636 --partition=boost_usr_prod --gres=gpu:1 --chdir=$RUN)
mode=${1:?}; shift
PT="set -e; module load python/3.11.7; unset PYTHONPATH; source $W/.venv/bin/activate; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore; cd $B6"
if [[ $mode == smoke ]]; then  # 1-epoch X pretrain (tag m7rz) -> one fold fit (new P01 seed 0, 2 epochs) -> pack not run
  p=$("${SB[@]}" --job-name=ftb-smoke-pt --cpus-per-task=8 --mem=40G --time=40 --output=$RUN/logs/%x-%j.out \
     --wrap "$PT; python -u laneB6/pretrain_sinc.py Z 0; cp runs/pretrain/sincshallow_ea-m7rz-s0.pt runs/pretrain/sincshallow_ea-m7rz-s0.map.json $ASSETS/runs/pretrain/")
  f=$("${SB[@]}" --job-name=ftb-smoke-fit --cpus-per-task=4 --mem=16G --time=30 --dependency=afterok:$p --kill-on-invalid-dep=yes --output=$RUN/logs/%x-%j.out \
     --wrap "CAP250_SMOKE=1 bash $REL/cleanslate/cap250/slurm.sh $OV/ftb_fit.py m7rz new P01 0 $RUN/smoke_fit_P01_s0")
  echo "smoke pt=$p fit=$f"; exit 0
fi
if [[ $mode == pretrain ]]; then  # array 0-5 = X seeds, 6-11 = C seeds
  "${SB[@]}" --job-name=ftb-pt --cpus-per-task=8 --mem=40G --time=90 --array=0-11 --output=$RUN/logs/%x-%A_%a.out \
    --wrap "$PT; a=(X X X X X X C C C C C C); s=\$((SLURM_ARRAY_TASK_ID%6)); t=\$( [ \${a[SLURM_ARRAY_TASK_ID]} = X ] && echo m7rx || echo m7rc ); python -u laneB6/pretrain_sinc.py \${a[SLURM_ARRAY_TASK_ID]} \$s; cp runs/pretrain/sincshallow_ea-\$t-s\$s.pt runs/pretrain/sincshallow_ea-\$t-s\$s.map.json $ASSETS/runs/pretrain/"
  exit 0
fi
if [[ $mode == fit ]]; then  # array over arm(2) x fold(12) x seed(6) = 144; DEP=<pretrain job id> optional
  "${SB[@]}" --job-name=ftb-fit --cpus-per-task=4 --mem=16G --time=45 --array=0-143%60 ${DEP:+--dependency=afterok:$DEP} --output=$RUN/logs/%x-%A_%a.out \
    --wrap "i=\$SLURM_ARRAY_TASK_ID; arm=\$((i/72)); fold=\$(((i%72)/6)); seed=\$((i%6)); tags=(m7rx m7rc); armn=(X C); \
      line=\$(sed -n \$((fold+1))p $FTB_JOBS); kind=\$(echo \"\$line\" | cut -f1); held=\$(echo \"\$line\" | cut -f2); \
      bash $REL/cleanslate/cap250/slurm.sh $OV/ftb_fit.py \${tags[arm]} \$kind \$held \$seed $RUN/parts/\${armn[arm]}/\${kind}_\$held/s\$seed"
  exit 0
fi
if [[ $mode == pack ]]; then  # DEP=<fit job id>
  for arm in X C; do
    "${SB[@]}" --job-name=ftb-pack-$arm --cpus-per-task=2 --mem=16G --time=30 ${DEP:+--dependency=afterok:$DEP} --output=$RUN/logs/%x-%j.out \
      --wrap "$W/.venv312/bin/python $OV/ftb_pack.py $PKG $RUN/parts/$arm $RUN/pkgs $arm"
  done; exit 0
fi
stream() { "${SB[@]}" --job-name=$1 --cpus-per-task=4 --mem=16G --time=45 ${8:-} --output=$RUN/logs/%x-%j.out \
  --wrap "bash $REL/cleanslate/cap250/slurm.sh cleanslate/cap250/stream_fold.py --pkg $2 --held $3 --sessions $4 --orders $5 --data $CAP250_DATA --out $6"; }
for orders in "$@"; do  # mode stream; DEP=<pack job ids, colon separated> optional
  tag=o${orders%%,*}; D=$RUN/$tag; mkdir -p $D
  for arm in ${ARMS:-X C}; do
    mkdir -p $D/$arm; cp $SRC/hardcb/jobs.tsv $D/$arm/jobs.tsv; deps=""
    while IFS=$'\t' read -r kind held _ enrol ses; do
      sess=all; [[ $kind == ret ]] && sess=$ses
      J=$(stream ftb-$arm-$kind$held-$tag $RUN/pkgs/$arm/${kind}_$held/pkg $held $sess $orders $D/$arm/pred_${kind}_$held.npz 45 ${DEP:+--dependency=afterok:$DEP})
      deps+=":$J"
    done < $SRC/hardcb/jobs.tsv
    S=$("${SB[@]}" --job-name=ftb-score-$arm-$tag --cpus-per-task=2 --mem=8G --time=20 --dependency=afterok$deps --output=$RUN/logs/%x-%j.out \
      --wrap "bash $REL/cleanslate/cap250/slurm.sh cleanslate/harness/score.py $D/$arm --data $CAP250_DATA")
    echo "$arm $tag score $S"
  done
done
