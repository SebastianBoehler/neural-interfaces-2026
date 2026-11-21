#!/usr/bin/env bash
# In leonardo-fill-21nov: bash control/scripts/leonardo_fill/queue_grow.sh screen|held
set -euo pipefail
phase=${1:?}
[[ $phase == screen || $phase == held ]]
root=$PWD
source "$root/env.sh"
wave=${LF_GROW_WAVE:-growth}
case "$wave" in
    growth) last_screen=47; candidates='swap25|fbc12|ba_six' ;;
    roster) last_screen=95; candidates='triseed25|ba_seven|ba_eight' ;;
    *) exit 2 ;;
esac
batch="$root/src-grow/scripts/leonardo_fill/grow.sbatch"
control="$root/control/scripts/leonardo_fill/queue_grow.sh"
manifest="$root/grow/$phase-jobs.tsv"
exec 9>"$root/grow/$phase-queue.lock"
flock -n 9

submit() {
    local name=$1 id
    shift
    if [[ -f $manifest ]]; then
        id=$(awk -F '\t' -v name="$name" '$1 == name {print $2}' "$manifest")
        if [[ -n $id ]]; then
            printf '%s\n' "$id"
            return
        fi
    fi
    id=$(sbatch --parsable --job-name="lf-grow-$name" "$@")
    id=${id%%;*}
    [[ $id =~ ^[0-9]+$ ]]
    printf '%s\t%s\n' "$name" "$id" >> "$manifest"
    printf '%s\n' "$id"
}

if [[ $phase == screen ]]; then
    for fold in new_P01 ret_P04 ret_P10; do
        smoke=$(awk -F '\t' -v fold="$fold" '$1 == fold {print $2}' grow-smokes.tsv)
        smoke=${smoke%%;*}
        [[ $smoke =~ ^[0-9]+$ ]]
        grep -qx OK "grow/v3-$smoke.log"
        grep -qx "PASS exact zero and lead labels, changed banks, graph and full terminal path" \
            "logs/lf-grow-smoke-$fold-$smoke.out"
        [[ $(sacct -nX -P -j "$smoke" -o State | xargs) == COMPLETED ]]
    done
    prep=$(submit prepare --array=1-9%9 --time=00:20:00 --output='logs/%x-%A_%a.out' "$batch" prepare)
    time=$(submit timing --array=0-5%6 --time=00:35:00 --output='logs/%x-%A_%a.out' "$batch" timing)
    screen=$(submit screen --dependency="afterok:$prep" --kill-on-invalid-dep=yes --array="0-$last_screen%24" \
        --output='logs/%x-%A_%a.out' "$batch" screen)
    freeze=$(submit select --dependency="afterok:$screen:$time" --kill-on-invalid-dep=yes \
        --time=00:15:00 --output='logs/%x-%j.out' "$batch" select)
    submit release --account=AIFAC_P02_1636 --partition=boost_usr_prod --gres=gpu:1 --cpus-per-task=1 \
        --mem=2G --time=00:05:00 --dependency="afterok:$freeze" --kill-on-invalid-dep=yes \
        --output='logs/%x-%j.out' --wrap="bash $control held"
else
    decision="$root/grow/selection.json"
    wait_for=()
    if ! grep -Eq '^ "chosen": null,$' "$decision"; then
        grep -Eq "^ \"chosen\": \"($candidates)\",$" "$decision"
        array=$(submit held --array=0-95%32 --output='logs/%x-%A_%a.out' "$batch" held)
        wait_for=(--dependency="afterok:$array" --kill-on-invalid-dep=yes)
    fi
    submit report "${wait_for[@]}" --time=00:15:00 --output='logs/%x-%j.out' "$batch" report
fi
cat "$manifest"
