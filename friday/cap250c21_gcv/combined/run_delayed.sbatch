#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-c21gct-cpu
#SBATCH --output=c21gct-cpu-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_js321_20261117_returning_pair_cpu1
inner=$base/ni26_friday_returning_native_inner_v2_20261119
observer=$base/ni26_friday_returning_core2_native_v3_20261119
original=$base/ni26_friday_return_eye_native_v1_20261119
r=$base/ni26_friday_return_fbl2_native_v2_20261119
rtx=$base/ni26_friday_c21e_native_v2_20261119
shared=$base/ni26_friday_c20s_native_v1_20261119
owner=$base/ni26_friday_c20e_native_v4_20261119
cross=$base/ni26_cross411b_20261118
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
retirement=$base/ni26_friday_c21gt_retired_v1_20261119
covariance=$base/ni26_friday_c21gcv_rtx_v1_20261119
(cd "$retirement" && echo '714add28327fb564cf2bb7b00b7f5420b58402d15ab74a95fc9b9ca5668de96c  source.sha256' | sha256sum --quiet -c
 sha256sum --quiet -c source.sha256
 echo '6af1cb7229dae99834e15a1564b931d9f8610503722a510b10d245119dea3b12  live_gate.py' | sha256sum --quiet -c)
(cd "$covariance" && echo '40ce7409bfc3f86ba53729b7e84c3b2ae9dd99f27392c51bf419abcfdd1c498d  source.sha256' | sha256sum --quiet -c
 sha256sum --quiet -c source.sha256
 echo '24016a105a5efbdbd424738c2cf46b3126d6538f96fa42af1844d21e75e9fca9  shared_cov.py' | sha256sum --quiet -c)
(cd "$base" && sha256sum --quiet -c "$r/native_inputs.sha256")
for dir in "$observer" "$prior" "$inner" "$original" "$r" "$rtx" "$owner"; do
    (cd "$dir" && sha256sum --quiet -c source.sha256)
done
echo "f52e414494585fd2d6926e050dd2ac0462ebedc6c8f9981d16e19e6d6d02a3cb  $shared/mirror.py" | sha256sum --quiet -c
grep -Fx OK "$base/ni26_aux431_20261118/runs/aux431/replay.txt"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env "OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl,NI26_WS=$repo,PYTHONPATH=$original:$observer:$inner:$prior:$retirement:$covariance:$rtx:$shared:$owner:$cross:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding" \
  --bind "$base:$base:ro,$base:/assets:ro,$work:/work:rw" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B combined_delayed.py
