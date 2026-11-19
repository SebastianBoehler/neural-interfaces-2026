#!/bin/bash
# Submit from a Slurm login. All data inspection, training and inference run in Slurm.
set -euo pipefail
root=$1
arch=$2
tag=$3
corpora=$4
mode=$5
source="$root/cleanslate/teamA/run.sbatch"
stage="$root/$tag"
orders=${6:-71000,71001}
mkdir -p "$stage"
gpu=()
cpu=()
if [[ "${TEAM_A_CLUSTER:-galvani}" == leonardo ]]; then
    gpu=(--account=AIFAC_P02_1675 --partition=boost_usr_prod --gres=gpu:1)
    cpu=(--account=AIFAC_P02_1675 --partition=boost_usr_prod --gres=gpu:1 --time=00:08:00)
fi
submit() {
    sbatch --parsable --chdir="$stage" --output='%x-%A_%a.out' "$@"
}
if [[ "$mode" == smoke ]]; then
    submit "${gpu[@]}" -t 00:16:00 -J "csA-smoke-$arch" "$source" smoke "$arch" "$root" "$stage" "$corpora"
    exit
fi
if [[ "$mode" == wave ]]; then
    smoke=$7
    if [[ -e "$stage/jobs.tsv" ]]; then
        echo "$stage already launched" >&2
        exit 2
    fi
    people=(P01 P02 P03 P04 P05 P06 P07 P08 P09 P10)
    all=$(IFS=,; echo "${people[*]}")
    for held in "${people[@]}"; do
        roster=()
        for person in "${people[@]}"; do
            [[ "$person" == "$held" ]] || roster+=("$person")
        done
        printf 'new\t%s\t%s\t\tall\n' "$held" "$(IFS=,; echo "${roster[*]}")"
    done > "$stage/jobs.tsv"
    printf 'ret\tP04\t%s\tP04\t2,3\nret\tP10\t%s\tP10\t2,3\n' "$all" "$all" >> "$stage/jobs.tsv"
    external=$(submit "${gpu[@]}" -t 01:30:00 -J "csA-ext-$arch" -d "afterok:$smoke" --kill-on-invalid-dep=yes \
        "$source" external "$arch" "$root" "$stage" "$corpora")
    fits=$(submit "${gpu[@]}" -t 01:45:00 -J "csA-fit-$arch" -d "afterok:${external%%;*}" --kill-on-invalid-dep=yes --array=0-11%8 \
        "$source" fold "$arch" "$root" "$stage" "$corpora" "$orders")
    if (( ${#cpu[@]} )); then
        score=$(submit "${cpu[@]}" -J "csA-score-$arch" -d "afterok:${fits%%;*}" --kill-on-invalid-dep=yes \
            "$source" score "$arch" "$root" "$stage" "$corpora")
    else
        score=$(submit -J "csA-score-$arch" -d "afterok:${fits%%;*}" --kill-on-invalid-dep=yes \
            --export="ALL,CS_REL=$root,CS_RUN=$stage" "$root/cleanslate/harness/score.sbatch")
    fi
    printf 'external=%s fits=%s score=%s\n' "$external" "$fits" "$score" | tee "$stage/queue.txt"
    exit
fi
echo "Invalid queue mode $mode" >&2
exit 2
