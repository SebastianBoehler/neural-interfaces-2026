#!/bin/bash
# ./waitrun.sh TAG N [pattern]: wait until run TAG has N result files (or no job of it is left), then fetch and score it.
tag=$1; n=$2; here=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 1 90); do
  out=$("$here/gssh" "ls /mnt/lustre/work/gehler/gfh527/ni26_csB_20261120/$tag/run/dev_*.npz 2>/dev/null | wc -l; squeue -u \$USER -h -o %Z | grep -c /$tag/run" 2>/dev/null)
  have=$(echo "$out" | sed -n 1p); left=$(echo "$out" | sed -n 2p)
  if [ "${have:-0}" -ge "$n" ] || { [ -n "$left" ] && [ "$left" = 0 ]; }; then break; fi
  sleep 60
done
"$here/pull.sh" "$tag" "${3:-@base}"
