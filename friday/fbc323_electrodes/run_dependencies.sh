#!/bin/bash
# Keep production held until the two actual source/held jobs pass.
set -euo pipefail
source_smoke=${1:?Ferranti source smoke}
source_matrix=${2:?Ferranti source matrix}
held_smoke=${3:?Galvani held smoke}
held_matrix=${4:?Galvani held matrix}
for id in "$source_smoke" "$source_matrix" "$held_smoke" "$held_matrix"; do
    [[ $id =~ ^[0-9]+$ ]]
done
home_script=$(cd "$(dirname "$0")" && pwd -P)
target=/mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_electrodes_20261119
opts=(-o BatchMode=yes -o ConnectTimeout=6)
deadline=$((SECONDS + 21600))

state() {
    ssh "${opts[@]}" "$1" "sacct -nX -P -j $2 -o JobID,State,ExitCode"
}

wait_smoke() {
    local result
    while (( SECONDS < deadline )); do
        result=$(state "$1" "$2") || { sleep 30; continue; }
        if [[ "$result" == "$2|COMPLETED|0:0" ]]; then
            return
        fi
        if [[ "$result" =~ (FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ]]; then
            printf '%s smoke %s exited with %s\n' "$1" "$2" "$result" >&2
            return 1
        fi
        sleep 30
    done
    printf '%s smoke %s did not pass within the scheduled follow-up.\n' "$1" "$2" >&2
    return 1
}

wait_smoke mlcloud-ferranti "$source_smoke"
case "${5:-start}" in
    start)
        ssh "${opts[@]}" mlcloud-ferranti "cd ~/ni26/friday_s1c_electrodes_20261119 &&
            grep -q 'PASS F27 paired source smoke_0_P01' fit-$source_smoke*.out &&
            tar -cf - ./*smoke*.pt ./*smoke*.json" |
            ssh "${opts[@]}" mlcloud-galvani "cd '$target/checkpoints' && tar -xf -"
        ssh "${opts[@]}" mlcloud-galvani "scontrol release $held_smoke"
        printf 'Released F27 held smoke %s after source completion.\n' "$held_smoke"
        ;;
    resume)
        [[ "$(state mlcloud-galvani "$held_smoke")" == "$held_smoke|COMPLETED|0:0" ]]
        ;;
    *) exit 2 ;;
esac
wait_smoke mlcloud-galvani "$held_smoke"
ssh "${opts[@]}" mlcloud-galvani "cd '$target' && grep -q 'PASS F27 held P01 seed 0' held-$held_smoke*.out"
if [[ "${5:-start}" == start ]]; then
    ssh "${opts[@]}" mlcloud-ferranti "scontrol release $source_matrix"
    printf 'Released F27 source array %s after independent held smoke.\n' "$source_matrix"
else
    printf 'Resuming F27 checkpoint relay for released source array %s.\n' "$source_matrix"
fi

while (( SECONDS < deadline )); do
    bash "$home_script/relay.sh" "$source_matrix" "$held_matrix" || { sleep 30; continue; }
    jobs=$(state mlcloud-ferranti "$source_matrix") || { sleep 30; continue; }
    complete=$(awk -F'|' '$2=="COMPLETED" && $3=="0:0" {n++} END {print n+0}' <<< "$jobs")
    if [[ "$complete" -eq 20 ]]; then
        bash "$home_script/relay.sh" "$source_matrix" "$held_matrix"
        exit 0
    fi
    if [[ "$jobs" =~ (FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY) ]]; then
        printf 'F27 production needs diagnosis.\n%s\n' "$jobs" >&2
        exit 1
    fi
    sleep 30
done
printf '%s\n' 'F27 source matrix is unfinished at the follow-up deadline.' >&2
exit 1
