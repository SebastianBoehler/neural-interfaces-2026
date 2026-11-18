#!/bin/bash
# Move only finished checkpoint/receipt pairs between the two Slurm clusters.
set -euo pipefail
ferranti=mlcloud-ferranti
galvani=mlcloud-galvani
source_dir=ni26/friday_wave_20261118
target_dir=/mnt/lustre/work/gehler/gfh527/ni26_friday_wave_20261118/checkpoints
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
ssh -o BatchMode=yes "$ferranti" 'cd ~/ni26/friday_wave_20261118; for f in *.json; do
    [[ -f "$f" && -s "${f%.json}.pt" ]] || continue
    printf "%s\n%s\n" "$f" "${f%.json}.pt"
done' > "$scratch/files"
if [[ ! -s "$scratch/files" ]]; then
    printf '%s\n' 'No complete checkpoint pairs on Ferranti.'
    exit
fi
rsync -a --files-from="$scratch/files" "$ferranti:$source_dir/" "$scratch/"
ssh -o BatchMode=yes "$galvani" mkdir -p "$target_dir"
rsync -a --delay-updates --files-from="$scratch/files" "$scratch/" "$galvani:$target_dir/"
awk '/[.]pt$/{n++} END {printf "Galvani has %d relayed checkpoint pairs.\n",n}' "$scratch/files"
