#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s310-release
#SBATCH --output=logs/s310-release-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s310_entropy_sign_20261116/production_v2
source studies/16nov/followups/meta_adaptation/env.sh
export S310_S305=/mnt/lustre/work/gehler/gfh527/ni26_s305_converged_v2_20261116
export S310_S305_ASSEMBLER=/mnt/lustre/work/gehler/gfh527/ni26_s305_assembler_recover_v1_20261116
for f in source.sha256 pinned.sha256 cpu-src.sha256 s310-src.sha256 pilot-pinned.sha256; do sha256sum -c "$f" >/dev/null; done
pilot=../smoke_v2
test "$(sacct -X -n -P -j 3115415 -o State,ExitCode | head -1)" = 'COMPLETED|0:0'
grep -q 'CPU entropy-sign streams pass anchor/reset/zero' "$pilot/logs/s310-pilot-3115415.out"
for file in flatten.py produce.py; do cmp "studies/16nov/followups/entropy_sign/$file" "$pilot/studies/16nov/followups/entropy_sign/$file"; done
cmp studies/16nov/followups/meta_adaptation/adapt.py "$pilot/studies/16nov/followups/meta_adaptation/adapt.py"
"$PY" -B -I studies/16nov/followups/entropy_sign/verify_shard.py P01
active=$(squeue -h -r -u "$USER" -o '%i' | wc -l)
if (( active + 12 > 272 )); then
 retry=$(sbatch --parsable --begin=now+10minutes studies/16nov/followups/entropy_sign/release.sbatch)
 printf 'capacity %s; retry %s\n' "$active" "$retry" >> release-retries.log
 exit 0
fi
set -o noclobber
printf 'release %s\n' "$SLURM_JOB_ID" > release.intent
set +o noclobber
mkdir -p runs/s310/target
cp -n "$pilot/runs/s310/target/P01.npz" runs/s310/target/P01.npz
cmp "$pilot/runs/s310/target/P01.npz" runs/s310/target/P01.npz
set -o noclobber
printf 'CPU %s %s\n' "$(sha256sum runs/s310/target/P01.npz | cut -d' ' -f1)" "$(sha256sum s310-src.sha256 | cut -d' ' -f1)" > runs/s310/target/P01.cpu-route
set +o noclobber
array=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=1-9%2 studies/16nov/followups/entropy_sign/target_array.sbatch)
printf 'target %s\n' "$array" >> release.intent
assembly=$(sbatch --parsable --dependency=afterok:"$array" studies/16nov/followups/entropy_sign/assemble.sbatch)
printf 'assembly %s\n' "$assembly" >> release.intent
