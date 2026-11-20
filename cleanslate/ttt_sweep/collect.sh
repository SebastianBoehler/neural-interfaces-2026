#!/bin/bash
# usage: collect.sh STAGE [extra collect.py args]  (Slurm CPU job via srun-free sbatch --wait; login node only submits)
root=$PWD; st=${1:?}; shift
sbatch --wait --account=AIFAC_P02_1675 -p lrd_all_serial -c 2 --mem=16G -t 00:30:00 -J t2-collect-$st -o logs/collect-$st-%j.out --wrap "set -e; module load python/3.11.7; unset PYTHONPATH; source /leonardo_work/AIFAC_P02_1636/ni26/.venv/bin/activate; export CS_DATA=$root/data/proteus_windows.npz CS_OFFICIAL=/leonardo_work/AIFAC_P02_1636/ni26/2026-competition/tracks/bci_decoding; cd $root; python cleanslate/ttt_sweep/collect.py work/$st --stage $st $* && python cleanslate/ttt_sweep/promote.py work/$st/${st}_leaderboard.json work/$st/promote.json"
