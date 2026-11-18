#!/bin/bash
# Move only finished checkpoint/receipt pairs between the two Slurm clusters.
set -euo pipefail
ferranti=mlcloud-ferranti
galvani=mlcloud-galvani
case ${1:-wave} in
    wave) release=friday_wave_20261118 ;;
    time) release=friday_wave_time_20261118 ;;
    motor) release=friday_wave_motor_20261118 ;;
    raw) release=friday_wave_raw_20261118 ;;
    style) release=friday_wave_style_20261118 ;;
    norm) release=friday_wave_norm_20261118 ;;
    band) release=friday_wave_band_20261118 ;;
    s1c_train) release=friday_s1c_training_20261118 ;;
    *) printf 'Unknown release %s\n' "$1" >&2; exit 2 ;;
esac
source_dir="ni26/$release"
target_dir="/mnt/lustre/work/gehler/gfh527/ni26_$release/checkpoints"
scratch=$(mktemp -d)
trap 'find "$scratch" -depth -delete' EXIT
ssh -o BatchMode=yes "$ferranti" "cd ~/ni26/$release" '; for f in *.json; do
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
