#!/bin/bash
# Submit Lane D from $WORK/ni26 with replay sources and orders.py staged inside the release.
set -euo pipefail
account=${1:?Leonardo account}
release=${2:?absolute release path}
stock=${3:?absolute stock ZIP path}
root=$PWD
code=$release
[[ "$release" = /* && "$stock" = /* && "$root" != *" "* && "$release" != *" "* && "$stock" != *" "* ]]
test -r "$release/orders.py"
test -r "$release/source.sha256"
test -r "$stock"
test -r "$root/data/scherer2015/windows.npz"
python=$root/.venv312/bin/python
test -x "$python"
mkdir -p "$release"/{banks,logs}
common=(--parsable --account="$account" -p boost_usr_prod --gres=gpu:1 -c 4 --mem=24G -t 00:50:00)
runtime="set -e; cd '$release'; sha256sum --quiet -c source.sha256; export PYTHONPATH='$release:$root/2026-competition' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
replay="'$python' '$code/worker.py' '$release' '$root/data/scherer2015/windows.npz'"
submit() { sbatch "${common[@]}" "$@" | cut -d ';' -f1; }
smoke=$(submit -J ni26-lane-d-smoke -o "$release/logs/smoke-%j.out" \
    --wrap "$runtime; ln -s '$stock' '$release/stock.zip'; '$python' '$code/make.py' '$stock' '$release/zips'; $replay 40 smoke cuda:0; printf '%s\n' PASS > '$release/replay.ok'")
array=$(submit -J ni26-lane-d -o "$release/logs/run-%A_%a.out" --array=0-71%12 \
    --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
    --wrap "$runtime; grep -Fx PASS '$release/replay.ok'; $replay \$SLURM_ARRAY_TASK_ID run cuda:0")
collect=$(submit -J ni26-lane-d-collect -o "$release/logs/collect-%j.out" \
    --dependency="afterok:$array" --kill-on-invalid-dep=yes \
    --wrap "$runtime; '$python' '$code/collect.py' '$release' '$root/data/scherer2015/windows.npz'")
printf 'smoke=%s array=%s collector=%s\n' "$smoke" "$array" "$collect" | tee "$release/job_ids.txt"
