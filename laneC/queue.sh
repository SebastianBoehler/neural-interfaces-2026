#!/bin/bash
set -euo pipefail
root=$(cd ../neural-interfaces-2026 && pwd -P)
sbatch_id() {
    sbatch --parsable "$@" | cut -d ';' -f1
}
submit() {
    id=$(sbatch_id "$@")
    [[ "$id" =~ ^[0-9]+$ ]]
    printf '%s %s\n' "$id" "$*" >> jobs.tsv
    printf '%s\n' "$id"
}
sha256sum *.py train.sbatch > laneC.sha256
oracle=$(submit -A gehler -p cpu-galvani -c 1 --mem=3G -t 5 -J ni26-laneC-oracle -o oracle-%j.out \
  --wrap "OPENBLAS_NUM_THREADS=1 $root/.venv/bin/python oracle.py .. $root/data/proteus_windows.npz $root/runs/shuffle_perms.npz oracle.json")
canary=$(submit -t 00:15:00 train.sbatch canary)
external=$(submit --dependency=afterok:"$canary" --kill-on-invalid-dep=yes --array=0-3%4 train.sbatch external)
a=$(submit --dependency=afterok:"$canary" --kill-on-invalid-dep=yes --array=0-7%8 train.sbatch proteus)
bc=$(submit --dependency=afterok:"$external" --kill-on-invalid-dep=yes --array=8-24%12 train.sbatch proteus)
report=$(submit -A gehler -p cpu-galvani -c 2 --mem=4G -t 10 -J ni26-laneC-collect -o collect-%j.out \
  --dependency=afterok:"$a":"$bc" --kill-on-invalid-dep=yes --wrap "OPENBLAS_NUM_THREADS=1 $root/.venv/bin/python collect.py $root $PWD")
printf 'oracle=%s canary=%s external=%s A=%s BC_returning=%s collector=%s\n' "$oracle" "$canary" "$external" "$a" "$bc" "$report"
