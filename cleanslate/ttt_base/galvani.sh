#!/bin/bash
# ttt_base overflow on Galvani (account gehler, a100 partitions). Same modes as leo.sh; release root /mnt/lustre/work/gehler/gfh527/ni26_tttbase_20261120.
# usage: leo.sh smoke
#        leo.sh run TAG "TRAIN_ARGS" VARIANTS ORDERS   -> array 0-9 = held P01..P10: train on the other 9, stream-eval (held-new LOPO)
#        leo.sh eval TAG VARIANTS ORDERS               -> array 0-9: stream-eval existing work/TAG/pkg_Pxx (SUB=name for the output dir)
#        leo.sh mix TAG VARIANTS ORDERS SRC1 SRC2 ...  -> array 0-9: merge the members of work/SRCi/pkg_Pxx (merge.py, optional PICK=name,name) then stream-eval
# env: ARR (array spec), T (time limit), DEP (afterok job id)
set -euo pipefail
root=$PWD; mode=${1:?}; acc=gehler
common="set -e; export PATH=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin:\$PATH; export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1; cd $root; P=\$(printf 'P%02d' \$((SLURM_ARRAY_TASK_ID+1))); O=\$(python -c \"print(','.join(f'P{i:02}' for i in range(1,11) if f'P{i:02}'!='\$P'))\")"
g="--account=$acc -p ${PART:-a100-galvani,a100-preemptable-galvani} --gres=gpu:1 -c 8 --mem=40G --exclude=${EXCL:-galvani-cn122} ${DEP:+--dependency=afterok:$DEP}"
mkdir -p logs work
SE=cleanslate/ttt_base/ref/stream_eval.py; TR=cleanslate/ttt_base/train.py
case $mode in
smoke)
  sbatch --parsable $g -t 00:30:00 --array=0 -J tb-smoke -o logs/smoke-%A_%a.out --wrap "$common; \
    python -u $TR --people \$O --out work/smoke/pkg_\$P --epochs 2 --members shallow:0,fbc@hp7:0,fbc@hbx:0,fbc64:0,fbcst8:0,fbwide:0,eegnex:0,ifnet:0; \
    python -u $SE --pkg work/smoke/pkg_\$P --held \$P --orders 0 --variants A0,A1,A2h --out work/smoke/eval_\$P.npz" ;;
run)
  tag=${2:?}; targs=${3:?}; var=${4:?}; ord=${5:?}
  mkdir -p work/$tag
  sbatch --parsable $g -t ${T:-03:00:00} --array=${ARR:-0-9} -J tb-$tag -o logs/$tag-%A_%a.out --wrap "$common; \
    python -u $TR --people \$O --out work/$tag/pkg_\$P $targs; \
    python -u $SE --pkg work/$tag/pkg_\$P --held \$P --orders $ord --variants $var --out work/$tag/eval_\$P.npz" ;;
eval)
  tag=${2:?}; var=${3:?}; ord=${4:?}; sub=${SUB:-ev}
  sbatch --parsable $g -t ${T:-03:00:00} --array=${ARR:-0-9} -J tb-$tag-$sub -o logs/$tag-$sub-%A_%a.out --wrap "$common; \
    mkdir -p work/$tag/$sub; python -u $SE --pkg work/$tag/pkg_\$P --held \$P --orders $ord --variants $var --out work/$tag/$sub/eval_\$P.npz" ;;
mix)
  tag=${2:?}; var=${3:?}; ord=${4:?}; shift 4; srcs=""; for s in "$@"; do srcs="$srcs work/$s/pkg_\$P"; done
  mkdir -p work/$tag
  sbatch --parsable $g -t ${T:-03:00:00} --array=${ARR:-0-9} -J tb-$tag -o logs/$tag-%A_%a.out --wrap "$common; \
    python -u cleanslate/ttt_base/merge.py work/$tag/pkg_\$P $srcs ${PICK:+--pick $PICK}; \
    python -u $SE --pkg work/$tag/pkg_\$P --held \$P --orders $ord --variants $var --out work/$tag/eval_\$P.npz" ;;
esac
