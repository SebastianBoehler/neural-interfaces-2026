#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=03:00:00
#SBATCH --job-name=s21-relE-submit
# Submits NI-V release E (queue script) once the task cap has room (<=150 submitted tasks). Run from ni26_final_confirm_20261121.
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_final_confirm_20261121
until [ "$(squeue -u $USER -h -r | wc -l)" -le 150 ]; do sleep 60; done
bash fin_niv_queue_gen.sh E
