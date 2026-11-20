#!/bin/bash
# Galvani runner. REL=/mnt/lustre/work/gehler/gfh527/ni26_tttvalid_20261120 (run from REL)
#   run.sh train            -> job: base on all ten (pkg_all) + base for returning (pkg_ret)
#   run.sh scherer DEPJOB   -> array 0-8 (persons A..), 8 orders
#   run.sh ret DEPJOB       -> array 0-1 (P04,P10), 8 orders
set -euo pipefail
root=$PWD; mode=${1:?}; dep=${2:-}
common="set -e; export PATH=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin:\$PATH OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1; cd $root"
g="--account=gehler -p ${PART:-a100-galvani,a100-preemptable-galvani,2080-galvani,2080-preemptable-galvani} --gres=gpu:1 -c 8 --mem=40G --exclude=galvani-cn122 ${dep:+--dependency=afterok:$dep}"
all=P01,P02,P03,P04,P05,P06,P07,P08,P09,P10
mem=shallow:0,shallow:1,eegnet:0,eegnet:1,fbnet:0,fbnet:1
V=${V:-A0,A1,A2,A2h,A3a}; O=${O:-0,1,2,3,4,5,6,7}
mkdir -p logs work
case $mode in
train) sbatch --parsable $g -t 01:00:00 -J ttv-train -o logs/train-%j.out --wrap "$common; \
  python -u cleanslate/teamC/train.py --people $all --members $mem --epochs 60 --out work/pkg_all; \
  python -u cleanslate/teamC/train.py --people $all --members $mem --epochs 60 --exclude 'P04|2,P04|3,P10|2,P10|3' --out work/pkg_ret" ;;
scherer) mkdir -p work/scherer; sbatch --parsable $g -t 04:00:00 --array=0-8 -J ttv-sch -o logs/sch-%A_%a.out --wrap "$common; \
  H=\$(python -c \"print('ACDEFGHJL'[\$SLURM_ARRAY_TASK_ID])\"); \
  python -u cleanslate/ttt_valid/stream_valid.py --pkg work/pkg_all --mode scherer --held \$H --orders $O --variants $V --out work/scherer/eval_\$H.npz" ;;
ret) mkdir -p work/ret; sbatch --parsable $g -t 04:00:00 --array=0-1 -J ttv-ret -o logs/ret-%A_%a.out --wrap "$common; \
  H=\$(python -c \"print(['P04','P10'][\$SLURM_ARRAY_TASK_ID])\"); \
  python -u cleanslate/ttt_valid/stream_valid.py --pkg work/pkg_ret --mode ret --held \$H --orders $O --variants $V --out work/ret/eval_\$H.npz" ;;
esac
