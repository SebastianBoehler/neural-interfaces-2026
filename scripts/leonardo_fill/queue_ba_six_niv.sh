#!/usr/bin/env bash
# From the Galvani ba_six run directory, after both exact NI-V smokes.
set -euo pipefail
root=$PWD
case ${1:-ordinary} in
    ordinary) prefix=; variant=ba_six_fc_niv; source=src ;;
    turing) prefix=turing-; variant=ba_six_fc_niv_fp16_turing; source=turing-qual-source ;;
    *) exit 2 ;;
esac
manifest="${prefix}niv-jobs.tsv"
exec 9>"$root/${prefix}niv-queue.lock"
flock -n 9
test ! -e "$manifest"
for mode in new ret; do
    id=$(awk -v t="${prefix}niv-smoke-$mode" '$1 == t {print $2}' jobs.tsv)
    [[ $id =~ ^[0-9]+$ ]]
    [[ $(sacct -nX -P -j "$id" -o State | xargs) == COMPLETED ]]
done
candidate="$root/niv-results/$variant"
for arm in lead trial ret_lead_k5 ret_trial_k5; do
    test -s "$candidate/$arm/smoke_5.npz"
done
batch="$root/$source/scripts/leonardo_fill/galvani_ba_six.sbatch"
submit() {
    local tag=$1 id
    shift
    id=$(sbatch --parsable --job-name="bsfc-${prefix}niv-$tag" "$@")
    id=${id%%;*}
    [[ $id =~ ^[0-9]+$ ]]
    printf '%s\t%s\n' "$tag" "$id" >> "$manifest"
    printf '%s\n' "$id"
}
new=$(submit new --array=0-35%16 --time=01:20:00 --output='logs/%x-%A_%a.out' "$batch" "${prefix}niv-new")
ret=$(submit ret --array=0-8%9 --time=01:30:00 --output='logs/%x-%A_%a.out' "$batch" "${prefix}niv-ret")
submit report --dependency="afterok:$new:$ret" --kill-on-invalid-dep=yes --time=00:15:00 \
    --output='logs/%x-%j.out' "$batch" "${prefix}niv-report"
cat "$manifest"
