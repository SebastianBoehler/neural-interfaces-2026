#!/bin/bash
# Run on Galvani after staging the two immutable score releases.
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
for release in memory454 gain454; do
    cd "$base/ni26_${release}_20261118"
    if [[ -s jobs.txt ]]; then
        cat jobs.txt
        continue
    fi
    if [[ $release == memory454 ]]; then
        arms=(kernel covariance)
    else
        arms=(coherent scattered)
    fi
    smoke=$(sbatch --parsable --hold score.sbatch smoke "${arms[0]}")
    echo "smoke $smoke" | tee jobs.txt
    for kind in "${arms[@]}"; do
        for seed in 454 1454; do
            id=$(sbatch --parsable --hold --kill-on-invalid-dep=yes --dependency=afterok:"$smoke" score.sbatch "$kind" "$seed")
            echo "$kind $seed $id" | tee -a jobs.txt
        done
    done
    if [[ $release == gain454 ]]; then
        for seed in 454 1454; do
            kernel=$(awk -v s="$seed" '$1=="kernel" && $2==s {print $3}' "$base/ni26_memory454_20261118/jobs.txt")
            deps=$(awk -v s="$seed" '$2==s && ($1=="coherent" || $1=="scattered") {print $3}' jobs.txt | paste -sd: -)
            id=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$kernel:$deps" score.sbatch compare "$seed")
            echo "compare $seed $id" | tee -a jobs.txt
        done
    fi
done
