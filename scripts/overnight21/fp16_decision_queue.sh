#!/usr/bin/env bash
# Run from the isolated Galvani fp16 decision directory.
set -euo pipefail

exec 9>queue.lock
flock -n 9
test ! -e jobs.tsv
submit() {
    local tag=$1 id
    shift
    id=$(sbatch --parsable --job-name="ni26-fp16-$tag" --output='logs/%x-%A_%a.out' "$@")
    id=${id%%;*}
    [[ $id =~ ^[0-9]+$ ]]
    printf '%s\t%s\n' "$tag" "$id" >> jobs.tsv
    printf '%s\n' "$id"
}
batch=$PWD/fp16_decision.sbatch
new_smoke=$(submit new-smoke "$batch" smoke-new)
ret_smoke=$(submit ret-smoke "$batch" smoke-ret)
timing_smoke=$(submit timing-smoke --nodelist=galvani-cn111 "$batch" timing-smoke)

# Three waves of three people. Each person uses four two-order shards.
new_dep=$new_smoke
for wave in 0 1 2; do
    ids=()
    for person in "$((wave * 3))" "$((wave * 3 + 1))" "$((wave * 3 + 2))"; do
        first=$((person * 4))
        ids+=("$(submit "new-p$person" --array="$first-$((first + 3))" \
            --dependency="afterok:$new_dep" --kill-on-invalid-dep=yes "$batch" new)")
    done
    new_dep=$(IFS=:; printf '%s' "${ids[*]}")
done
ret=$(submit ret --array=0-8%6 --dependency="afterok:$ret_smoke" --kill-on-invalid-dep=yes "$batch" ret)
submit report --time=00:10:00 --dependency="afterok:$new_dep:$ret" --kill-on-invalid-dep=yes "$batch" report

for node in 111 114 115 116 122 123; do
    first=$(submit "cn$node-r1" --nodelist="galvani-cn$node" \
        --dependency="afterok:$timing_smoke" --kill-on-invalid-dep=yes "$batch" timing)
    submit "cn$node-r2" --nodelist="galvani-cn$node" --begin=2026-11-21T16:55:00 \
        --dependency="afterok:$first" --kill-on-invalid-dep=yes "$batch" timing
done
cat jobs.tsv
