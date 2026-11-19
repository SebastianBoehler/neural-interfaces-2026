#!/bin/bash
# Relay only successful paired native checkpoints. Final jobs wait for actual export smoke.
set -euo pipefail
fit_smoke=${1:?Ferranti fit smoke}
fit_final=${2:?Ferranti fit array}
rtx_smoke=${3:?Galvani export smoke}
rtx_final=${4:?Galvani export array}
for job in "$fit_smoke" "$fit_final" "$rtx_smoke" "$rtx_final"; do
    [[ $job =~ ^[0-9]+$ ]]
done
opts=(-o BatchMode=yes -o ConnectTimeout=6)
source='~/ni26/friday_s1c_motor_external_20261119'
target=/mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_motor_external_20261119
deadline=$((SECONDS + 14400))

wait_ok() {
    local state
    while (( SECONDS < deadline )); do
        state=$(ssh "${opts[@]}" "$1" "sacct -nX -P -j $2 -o JobID,State,ExitCode") || { sleep 20; continue; }
        if [[ "$state" == "$2|COMPLETED|0:0" ]]; then
            return
        fi
        if [[ "$state" =~ (FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) ]]; then
            printf '%s failed %s: %s\n' "$1" "$2" "$state" >&2
            return 1
        fi
        sleep 20
    done
    printf '%s %s exceeded monitor deadline\n' "$1" "$2" >&2
    return 1
}

relay() {
    local id=$1 tag=$2
    local files=() arm
    for arm in hard detached joint; do
        files+=("${arm}_$tag.pt" "${arm}_$tag.json")
    done
    ssh "${opts[@]}" mlcloud-ferranti "set -e; cd $source; grep -q 'PASS F32e paired source $tag' fit-$id*.out; tar -cf - ${files[*]}" |
        ssh "${opts[@]}" mlcloud-galvani "cd '$target/checkpoints' && tar --keep-old-files -xf -"
}

wait_ok mlcloud-ferranti "$fit_smoke"
relay "$fit_smoke" smoke_0_all_source
ssh "${opts[@]}" mlcloud-galvani "scontrol release $rtx_smoke"
printf 'Released RTX smoke %s\n' "$rtx_smoke"

wait_ok mlcloud-galvani "$rtx_smoke"
ssh "${opts[@]}" mlcloud-galvani "cd '$target' && grep -q 'PASS F32e native worker seed 0 receipt' export-$rtx_smoke*.out"
ssh "${opts[@]}" mlcloud-ferranti "scontrol release $fit_final"
printf 'Released Ferranti fits %s\n' "$fit_final"

for seed in 0 1; do
    wait_ok mlcloud-ferranti "${fit_final}_$seed"
    relay "$fit_final" "${seed}_all_source"
    ssh "${opts[@]}" mlcloud-galvani "scontrol release ${rtx_final}_$seed"
    printf 'Released RTX export %s for seed %s\n' "$rtx_final" "$seed"
done
for seed in 0 1; do
    wait_ok mlcloud-galvani "${rtx_final}_$seed"
done
printf 'Completed both native F32e exports.\n'
