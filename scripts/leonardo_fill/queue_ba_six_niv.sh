#!/usr/bin/env bash
# From the Galvani ba_six run directory, after both exact NI-V smokes.
set -euo pipefail
root=$PWD
exec 9>"$root/niv-queue.lock"
flock -n 9
test ! -e niv-jobs.tsv
for mode in new ret; do
    id=$(awk -v t="niv-smoke-$mode" '$1 == t {print $2}' jobs.tsv)
    [[ $id =~ ^[0-9]+$ ]]
    [[ $(sacct -nX -P -j "$id" -o State | xargs) == COMPLETED ]]
done
candidate="$root/niv-results/ba_six_fc_niv"
for arm in lead trial ret_lead_k5 ret_trial_k5; do
    test -s "$candidate/$arm/smoke_5.npz"
done
batch="$root/src/scripts/leonardo_fill/galvani_ba_six.sbatch"
submit() {
    local tag=$1 id
    shift
    id=$(sbatch --parsable --job-name="bsfc-niv-$tag" "$@")
    id=${id%%;*}
    [[ $id =~ ^[0-9]+$ ]]
    printf '%s\t%s\n' "$tag" "$id" >> niv-jobs.tsv
    printf '%s\n' "$id"
}
new=$(submit new --array=0-35%16 --time=01:20:00 --output='logs/%x-%A_%a.out' "$batch" niv-new)
ret=$(submit ret --array=0-8%9 --time=01:30:00 --output='logs/%x-%A_%a.out' "$batch" niv-ret)
submit report --dependency="afterok:$new:$ret" --kill-on-invalid-dep=yes --time=00:15:00 \
    --output='logs/%x-%j.out' "$batch" niv-report
cat niv-jobs.tsv
