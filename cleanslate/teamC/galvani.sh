#!/bin/bash
# Team C on Galvani (a100-galvani / a100-preemptable-galvani). Run from the release root (a copy of the repo paths used).
# usage: leo.sh smoke
#        leo.sh run TAG "TRAIN_ARGS" VARIANTS ORDERS     -> array 0-9 = held P01..P10: train on the other 9, stream-eval
#        galvani.sh ret TAG "TRAIN_ARGS" VARIANTS ORDERS -> returning dev: train on all but P04/P10 sessions 2-3, stream those
#        leo.sh eval TAG VARIANTS ORDERS                -> array 0-9: stream-eval the existing TAG packages (no training)
set -euo pipefail
root=$PWD; mode=${1:?}
common="set -e; export PATH=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin:\$PATH; export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1; cd $root; P=\$(printf 'P%02d' \$((SLURM_ARRAY_TASK_ID+1))); O=\$(python -c \"print(','.join(f'P{i:02}' for i in range(1,11) if f'P{i:02}'!='\$P'))\")"
g="--account=gehler -p ${PART:-a100-galvani,a100-preemptable-galvani,a100-fat-galvani,2080-galvani,2080-preemptable-galvani} --gres=gpu:1 -c 8 --mem=40G --exclude=${EXCL:-galvani-cn122} ${DEP:+--dependency=afterok:$DEP}"
mkdir -p logs work
case $mode in
smoke)
  sbatch --parsable $g -t 00:20:00 --array=0 -J tc-smoke -o logs/smoke-%A_%a.out --wrap "$common; \
    python -u cleanslate/teamC/train.py --people \$O --out work/smoke/pkg_\$P --epochs 2; \
    python -u cleanslate/teamC/stream_eval.py --pkg work/smoke/pkg_\$P --held \$P --orders 0 --variants A0,A1,A2,A2h --out work/smoke/eval_\$P.npz" ;;
run)
  tag=${2:?}; targs=${3:?}; var=${4:?}; ord=${5:?}
  mkdir -p work/$tag
  sbatch --parsable $g -t ${T:-02:00:00} --array=${ARR:-0-9} -J tc-$tag -o logs/$tag-%A_%a.out --wrap "$common; \
    python -u cleanslate/teamC/train.py --people \$O --out work/$tag/pkg_\$P $targs; \
    python -u cleanslate/teamC/stream_eval.py --pkg work/$tag/pkg_\$P --held \$P --orders $ord --variants $var --out work/$tag/eval_\$P.npz" ;;
eval)
  tag=${2:?}; var=${3:?}; ord=${4:?}; sub=${SUB:-ev}
  sbatch --parsable $g -t ${T:-02:00:00} --array=${ARR:-0-9} -J tc-$tag-$sub -o logs/$tag-$sub-%A_%a.out --wrap "$common; \
    mkdir -p work/$tag/$sub; python -u cleanslate/teamC/stream_eval.py --pkg work/$tag/pkg_\$P --held \$P --orders $ord --variants $var --out work/$tag/$sub/eval_\$P.npz" ;;
ret)
  tag=${2:?}; targs=${3:?}; var=${4:?}; ord=${5:?}; all=P01,P02,P03,P04,P05,P06,P07,P08,P09,P10
  mkdir -p work/$tag
  sbatch --parsable $g -t ${T:-02:00:00} --array=0 -J tc-$tag -o logs/$tag-%A_%a.out --wrap "$common; \
    python -u cleanslate/teamC/train.py --people $all --exclude 'P04|2,P04|3,P10|2,P10|3' --out work/$tag/pkg_ret $targs; \
    for h in P04 P10; do python -u cleanslate/teamC/stream_eval.py --pkg work/$tag/pkg_ret --held \$h --sessions 2,3 --orders $ord --variants $var --out work/$tag/eval_\$h.npz; done" ;;
esac
