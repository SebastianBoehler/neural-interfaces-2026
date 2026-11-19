#!/bin/bash
# Prepare the same external experiment on Leonardo A100. Run from the staged repo.
set -euo pipefail
account=${1:?authorized Leonardo Slurm account}
release=${2:?absolute unused release directory}
stock=${3:?absolute signed cap250 ZIP}
manifest=${4:?absolute signed Scherer run manifest}
root=$PWD
code=$root/friday/lane_b_shuffle
python=$root/.venv312/bin/python
public=$root/data/proteus_windows.npz
external=$root/data/scherer2015/windows.npz
[[ $release = /* && $stock = /* && $manifest = /* ]]
test -x "$python"
test -r "$stock" && test -r "$manifest" && test -r "$public" && test -r "$external"
mkdir "$release"
mkdir "$release"/{zips,logs,out}
ln -s "$stock" "$release/zips/stock.zip"

quote() { printf '%q ' "$@"; }
env_setup="set -e; export $(quote "PYTHONPATH=$root/niv" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)"
test_run=$(quote "$python" -B "$code/smoke.py" smoke "$release/zips" "$external" "$manifest" "$public" cuda:0)
check=$(quote "$python" -B "$root/replay/check.py")
make=$(quote "$python" -B "$code/make.py" "$stock" "$release/zips")
run=$(quote "$python" -B "$code/external.py" "$release/zips" "$external" "$manifest")
out_prefix=$(printf '%q' "$release/out/")
summary=$(quote "$python" -B "$code/collect.py" "$external" "$manifest" "$release/out" "$release/external.json")
submit() { sbatch --parsable --account="$account" --kill-on-invalid-dep=yes "$@" | cut -d ';' -f1; }
gpu=(--partition=boost_usr_prod --gres=gpu:1 --cpus-per-task=4 --mem=24G)
smoke=$(submit "${gpu[@]}" -t 00:35:00 -J ni26-b-smoke -o "$release/logs/smoke-%j.out" \
    --wrap "$env_setup; $check > $(quote "$release/replay.log"); grep -Fx OK $(quote "$release/replay.log"); $make; $test_run")
array=$(submit "${gpu[@]}" -t 01:20:00 -J ni26-b-external -o "$release/logs/external-%A_%a.out" \
    --dependency=afterok:"$smoke" --array=0-17%9 \
    --wrap "$env_setup; i=\$((SLURM_ARRAY_TASK_ID * 8)); end=\$((i + 8)); while [ \$i -lt \$end ]; do
        $run $out_prefix\$i.npz \$i; i=\$((i + 1)); done")
collect=$(submit --partition=lrd_all_serial --cpus-per-task=2 --mem=8G -t 00:12:00 \
    -J ni26-b-collect -o "$release/logs/collect-%j.out" --dependency=afterok:"$array" --wrap "$env_setup; $summary")
printf 'smoke=%s array=%s collector=%s\n' "$smoke" "$array" "$collect" | tee "$release/jobs.txt"
