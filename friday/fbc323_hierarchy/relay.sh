#!/bin/bash
# Relay authenticated finished F19 folds and release the corresponding held elements.
set -euo pipefail
source_job=${1:?Ferranti source array ID}
held_job=${2:?Galvani held array ID}
[[ $source_job =~ ^[0-9]+$ && $held_job =~ ^[0-9]+$ ]]
ferranti=mlcloud-ferranti
galvani=mlcloud-galvani
target=/mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_hierarchy_20261118
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=6)
finished=$(ssh "${ssh_opts[@]}" "$ferranti" "sacct -nX -P -j $source_job -o JobID,State,ExitCode")
seen=$(ssh "${ssh_opts[@]}" "$galvani" "cd '$target/checkpoints' && find . -maxdepth 1 -name 'release_$held_job-*' -print")
files=()
ids=()
while IFS='|' read -r job status code; do
    [[ $job =~ ^${source_job}_([0-9]+)$ && $status == COMPLETED && $code == 0:0 ]] || continue
    id=${BASH_REMATCH[1]}
    [[ "$seen" != *"./release_$held_job-$id."* ]] || continue
    (( id < 20 ))
    printf -v tag '%d_P%02d' "$((id/10))" "$((id%10+1))"
    for arm in hard jitter hierarchy hierarchy_jitter; do
        files+=("${arm}_$tag.pt" "${arm}_$tag.json")
    done
    ids+=("$id")
done <<< "$finished"
if (( ${#ids[@]} == 0 )); then
    exit 0
fi
ssh "${ssh_opts[@]}" "$ferranti" "cd ~/ni26/friday_s1c_hierarchy_20261118 && tar -cf - ${files[*]}" |
    ssh "${ssh_opts[@]}" "$galvani" "cd '$target/checkpoints' && tar -xf -"
ssh "${ssh_opts[@]}" "$galvani" "set -e; cd '$target/checkpoints'; for i in ${ids[*]}; do
    scontrol release ${held_job}_\$i
    printf '%s\n' '$source_job' > release_$held_job-\$i.done
done"
printf 'Released F19 held folds %s\n' "${ids[*]}"
