#!/bin/bash
# s283/s286/s287 grids on Galvani (run on Galvani in the work dir; only sbatch). Lanes in total <= 24: teacher %6, coral %10, B1 %4, dist %8 (after the teachers).
# CMD uses the 2080 partitions; PART overrides. Usage: PART=.. studies/15nov/grid283/submit.sh teacher|coral|b1|dist|combined "FAMS (combined only)"
set -euo pipefail
export PART=${PART:-2080-galvani,2080-preemptable-galvani}
C="fb_coral3,fb_coral10,fb_coral30,fb_coral100"; for b in fbl sinc eegn; do for w in 1 3 10 30 100; do C="$C,${b}_coral$w"; done; done  # fb w1 exists (s285)
case $1 in
teacher) sbatch --parsable --account=gehler -p $PART --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00 --requeue ${DEP:-} --array=0-29%6 -J g283-teacher -o logs/teacher-%A_%a.out --wrap "
export PATH=\$PWD/.venv/bin:\$PATH NI26_WS=\$PWD HF_HOME=\$PWD/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
python -c 'import torch; assert torch.cuda.is_available()' || exit 1
python studies/15nov/grid283/teacher.py \$SLURM_ARRAY_TASK_ID; rc=\$?; echo EXIT \$rc; exit \$rc";;
coral) scripts/cloud/galv_array.sh g286coral 06:00:00 "FAMS=$C bash studies/15nov/grid283/cell.sh" "0-227%10";;
b1) scripts/cloud/galv_array.sh g287b1 06:00:00 "FAMS=fbl_am25,fbl_am50,fbl_am100,sinc_am25,sinc_am50,sinc_am100 bash studies/15nov/grid283/cell.sh" "0-71%4";;
dist) scripts/cloud/galv_array.sh g283dist 06:00:00 "FAMS=fb_dist1,fb_dist2,fb_dist4,eegn_dist1,eegn_dist2,eegn_dist4 bash studies/15nov/grid283/cell.sh" "0-71%8";;
combined) N=$(echo $2 | tr ',' '\n' | wc -l); scripts/cloud/galv_array.sh g287comb 06:00:00 "FAMS=$2 bash studies/15nov/grid283/cell.sh" "0-$((12*N-1))%8";;
esac
