#!/bin/bash
# Submit a team B dev run from the Mac: ./submit.sh TAG "ORDERS" "VARIANTS" "ENV(NAME=VAL;...)" [people] [modes] [extra sbatch args]
# rsyncs cleanslate/ to a per-tag snapshot so queued jobs never see later edits.
set -euo pipefail
tag=$1; orders=${2:-71000,71001}; variants=${3:-base}; env=${4:-}; people=${5:-P01,P02,P03,P04,P05,P06,P07,P08,P09,P10}; modes=${6:-new}
shift $(( $# < 6 ? $# : 6 ))
base=/mnt/lustre/work/gehler/gfh527; rel=$base/ni26_csB_20261120/$tag
here=$(cd "$(dirname "$0")/.." && pwd)
ssh mlcloud-galvani "mkdir -p $rel/cleanslate $rel/run && test ! -e $rel/run/jobs.tsv"
rsync -a --exclude __pycache__ --exclude 'ext' "$here/harness" "$here/teamB" mlcloud-galvani:$rel/cleanslate/
rows=""; for m in ${modes//,/ }; do for p in ${people//,/ }; do
  if [ "$m" = new ] || [ "$p" = P04 ] || [ "$p" = P10 ]; then rows+="$m\t$p\n"; fi; done; done
n=$(printf "$rows" | wc -l | tr -d ' ')
ssh mlcloud-galvani "printf '$rows' > $rel/run/jobs.tsv && cd $rel/run && sbatch --parsable --array=0-$((n-1)) \
  --partition=a100-galvani,a100-preemptable-galvani,2080-galvani,2080-preemptable-galvani --gres=gpu:1 --time=04:00:00 $* \
  --export=ALL,CS_REL=$rel,CS_RUN=$rel/run,CS_ORDERS='${orders//,/;}',CS_VARIANTS='$variants',CS_ENV_SEMI='$env' $rel/cleanslate/teamB/dev.sbatch"
