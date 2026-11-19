#!/bin/bash
# Lane E: stage and queue everything on Galvani in one go (needs an eduVPN + 2FA ssh session; run from the repo root).
#   lane_e/release_galvani.sh ZIP_DIR        (ZIP_DIR is unused: arms are built on the cluster by build_arms.sbatch from the stock archive, members/ and submission.py)
# Submits only Slurm jobs (never work on the login node): A100 + RTX timing, NI-V B-new scoring of Ea Eb Ec Ed (36 tasks each),
# E0 (tasks 0-3) + bitwise null check against cap250, and on-cluster collectors writing out/report_laneE_<arm>_vs_cap250.json
# (paired delta, people-and-order intervals, null from cap250_seed1). The cap250 / cap250_seed1 reference jobs are the NI-V
# owner's (3136444 / 3136445, verified below by their submit lines); the collectors wait for them.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(dirname "$here"); Z=${1:-}
B=/mnt/lustre/work/gehler/gfh527; W=$B/ni26_lane_e_capacity_20261119; R=$B/ni26_niv_v1_20261119; X="--exclude=galvani-cn122,galvani-cn116"
ssh mlcloud-galvani "mkdir -p $W/code/lane_e $W/code/submissions/lane_e_capacity $W/logs $W/out $W/tmp $R/laneE"
rsync -a --exclude __pycache__ "$here"/ mlcloud-galvani:$W/code/lane_e/
rsync -a "$repo"/submissions/lane_e_capacity/submission.py mlcloud-galvani:$W/code/submissions/lane_e_capacity/
rsync -a "$here"/timing.sbatch "$here"/build_arms.sbatch mlcloud-galvani:$W/
rsync -a "$here"/null_check.sbatch mlcloud-galvani:$R/
rsync -a "$here"/null_check.py mlcloud-galvani:$R/laneE/
ssh mlcloud-galvani "set -euo pipefail
cd $W/code && ln -sfn $B/neural-interfaces-2026/data data && ln -sfn $B/neural-interfaces-2026/2026-competition 2026-competition
cd $W && rm -f zips/*.zip && find code/scripts code/experiments code/studies code/lane_e code/submissions -type f -name '*.py' | sort | xargs sha256sum > code.sha256
[ \$(sacct -j 3136444 -n -o SubmitLine%150 | grep -c 'v66_cp1_cap250.zip cap250') -ge 1 ] || { echo 'cap250 reference job 3136444 is not the cap250 tag'; exit 1; }
b=\$(sbatch --parsable build_arms.sbatch); echo build \$b
D=--dependency=afterok:\$b
t1=\$(sbatch --parsable \$D -p a100-galvani --gres=gpu:a100:1 timing.sbatch); t2=\$(sbatch --parsable \$D -p 2080-galvani --gres=gpu:rtx2080ti:1 timing.sbatch)
echo timing A100 \$t1 RTX \$t2
cd $R
for a in Ea Eb Ec Ed; do
  j=\$(sbatch --parsable $X \$D --array=0-35 score.sbatch laneE_\$a.zip laneE_\$a)
  c=\$(sbatch --parsable --dependency=afterany:\$j:3136444:3136445 collect.sbatch laneE_\$a cap250 cap250_seed1)
  echo \$a score \$j collector \$c
done
j0=\$(sbatch --parsable $X \$D --array=0-3 score.sbatch laneE_E0.zip laneE_E0)
n=\$(sbatch --parsable --dependency=afterany:\$j0:3136444 null_check.sbatch laneE_E0 cap250)
echo E0 score \$j0 null-check \$n"
