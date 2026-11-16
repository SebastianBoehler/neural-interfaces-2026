#!/bin/bash
# Resume after stage_training.sh stopped before copying the three .pt files.
set -euo pipefail
root=/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/train
old=/mnt/lustre/work/gehler/gfh527/ni26_grid283
source_root=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
revision=dc2a075c287bb2f6c04ee5875bd79535a0f7dba6
cd "$root"
test -f data/proteus_windows.npz
test -f runs/shuffle_perms.npz
test ! -e inputs.sha256
test -z "$(find runs/pretrain runs/diag/reve283 -type f -print -quit)"
cp "$old"/runs/pretrain/*.pt runs/pretrain/
cp "$old"/runs/diag/reve283/teacher_s*.npz runs/diag/reve283/
mkdir -p 2026-competition/tracks/bci_decoding
test ! -e 2026-competition/tracks/bci_decoding/benchmark_utils
cp -R "$source_root/2026-competition/benchmark_utils" 2026-competition/tracks/bci_decoding/
model="hf/hub/models--brain-bzh--reve-base/snapshots/$revision"
mkdir -p "$model"
test ! -e "$model/model.safetensors"
cp -L "$old/$model/model.safetensors" "$model/model.safetensors"
test "$(sha256sum "$model/model.safetensors" | cut -d' ' -f1)" = 8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe
test "$(find runs/diag/reve283 -maxdepth 1 -name 'teacher_s*.npz' | wc -l)" -eq 30
sha256sum studies/15nov/grid283/teacher.py studies/15nov/fm280/reve_fm.py studies/12nov/ttt5.py studies/13nov/ttt7.py experiments/deep.py experiments/protocol.py data/proteus_windows.npz runs/shuffle_perms.npz "$model/model.safetensors" > inputs.sha256
printf 'private teacher workspace prepared: %s\n' "$PWD"
