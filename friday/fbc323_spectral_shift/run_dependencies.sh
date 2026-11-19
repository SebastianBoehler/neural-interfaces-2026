#!/bin/bash
# Release exact source fits only after both real F37 smokes pass.
set -euo pipefail
source_smoke=${1:?Ferranti source smoke}
source_array=${2:?Ferranti source array}
held_smoke=${3:?Galvani held smoke}
held_array=${4:?Galvani held array}
relay=${5:?path to materialized F37 relay.sh}
for id in "$source_smoke" "$source_array" "$held_smoke" "$held_array"; do
    [[ $id =~ ^[0-9]+$ ]]
done
opts=(-o BatchMode=yes -o ConnectTimeout=6)
ferranti=mlcloud-ferranti
galvani=mlcloud-galvani
target=/mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_spectral_shift_20261119
due=$((SECONDS + 21600))

state() {
    ssh "${opts[@]}" "$1" "sacct -nX -P -j $2 -o JobID,State,ExitCode"
}

wait_smoke() {
    local result
    while (( SECONDS < due )); do
        result=$(state "$1" "$2") || { sleep 30; continue; }
        if [[ "$result" == "$2|COMPLETED|0:0" ]]; then
            return
        fi
        if [[ "$result" =~ (FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ]]; then
            printf '%s smoke %s: %s\n' "$1" "$2" "$result" >&2
            return 1
        fi
        sleep 30
    done
    return 1
}

wait_smoke "$ferranti" "$source_smoke"
ssh "${opts[@]}" "$ferranti" "cd ~/ni26/friday_s1c_spectral_shift_20261119 &&
    grep -q 'PASS F37 paired source smoke_0_P01' fit-$source_smoke*.out &&
    tar -cf - hard_smoke_0_P01.{pt,json} beta_smoke_0_P01.{pt,json} alpha_smoke_0_P01.{pt,json}" |
    ssh "${opts[@]}" "$galvani" "cd '$target/checkpoints' && tar -xf -"
ssh "${opts[@]}" "$galvani" "scontrol release $held_smoke"
wait_smoke "$galvani" "$held_smoke"
ssh "${opts[@]}" "$galvani" "cd '$target' && grep -q 'PASS F37 held P01 seed 0' held-$held_smoke*.out"
ssh "${opts[@]}" "$ferranti" "scontrol release $source_array"
printf 'True source and held smokes passed. Released source array %s.\n' "$source_array"

while (( SECONDS < due )); do
    bash "$relay" "$source_array" "$held_array" || { sleep 30; continue; }
    jobs=$(state "$ferranti" "$source_array") || { sleep 30; continue; }
    complete=$(awk -F'|' '$2=="COMPLETED" && $3=="0:0" {n++} END {print n+0}' <<< "$jobs")
    if [[ "$complete" -eq 20 ]]; then
        bash "$relay" "$source_array" "$held_array"
        exit 0
    fi
    if [[ "$jobs" =~ (FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY) ]]; then
        printf 'F37 source array needs diagnosis.\n%s\n' "$jobs" >&2
        exit 1
    fi
    sleep 30
done
printf 'F37 source array still incomplete at the relay deadline.\n' >&2
exit 1
