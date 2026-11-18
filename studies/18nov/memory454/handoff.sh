#!/bin/bash
# Workstation relay. Every copied receipt is verified again by the cluster scorer.
set -euo pipefail
f=mlcloud-ferranti
g=mlcloud-galvani
cache=${TMPDIR:-/tmp}/ni26-memory454-relay
mkdir -p "$cache"
opts=(-n -o BatchMode=yes -o ConnectTimeout=10)
export RSYNC_RSH='ssh -o BatchMode=yes -o ConnectTimeout=10'
for release in memory454 gain454; do
    project=/mnt/lustre/work/gehler/gfh527/ni26_${release}_20261118
    mkdir -p "$cache/$release"
    cd "$cache/$release"
    if [[ $release == memory454 ]]; then first=kernel; else first=coherent; fi
    ssh "${opts[@]}" "$g" "cat $project/jobs.txt" > jobs
    rsync -a "$f:ni26/$release/source.sha256" actor_source.sha256
    rsync -a actor_source.sha256 "$g:$project/"
    while read -r kind seed id; do
        if [[ $kind == compare ]]; then continue; fi
        if [[ $kind == smoke ]]; then
            id=$seed
            pattern=${first}_454_P01
            n=1
        else
            pattern=${kind}_${seed}_P
            n=10
        fi
        if [[ -e sent_$id ]]; then continue; fi
        count=$(ssh "${opts[@]}" "$f" "cd ~/ni26/$release && find . -maxdepth 1 -name '${pattern}*.json' | wc -l")
        if [[ $count -ne $n ]]; then continue; fi
        rsync -a --include="${pattern}*.npz" --include="${pattern}*.json" --exclude='*' "$f:ni26/$release/" ./
        rsync -a --include="${pattern}*.npz" --include="${pattern}*.json" --exclude='*' ./ "$g:$project/"
        ssh "${opts[@]}" "$g" "reason=\$(squeue -h -j $id -o %r); if test \"\$reason\" = JobHeldUser; then scontrol release $id; fi; test \"\$(squeue -h -j $id -o %r)\" != JobHeldUser"
        touch sent_"$id"
        echo "$(date -u) released $id after $n $pattern receipts"
    done < jobs
done
if [[ $(find "$cache" -name 'sent_*' | wc -l) -eq 10 ]] && launchctl list org.ni26.memory454.relay >/dev/null 2>&1; then
    launchctl remove org.ni26.memory454.relay
fi
