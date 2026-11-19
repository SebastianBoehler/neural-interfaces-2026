#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-c25pdet
#SBATCH --output=repeat-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
pair=$base/ni26_friday_frozen_rtx_20261118
other=$base/ni26_friday_frozen_rtx_nine_20261118
known=$base/ni26_native366_exact_ea_normal_v1_20261117/reference_normal.json
rtx_known=$base/ni26_friday_c5r_rtx_v1_20261119/smoke.json
native=$base/ni26_friday_c21e_native_v2_20261119
shared=$base/ni26_friday_c20s_native_v1_20261119
owner=$base/ni26_friday_c20e_native_v4_20261119
f32=$base/ni26_friday_f32x_external_cpu_v2_20261119
stock=$base/ni26_s318_coral_fbl_20261117/runs/sealed/final_zips/v66_cp1_cap250.zip
drift=$base/ni26_friday_c21gpack_drift_v1_20261119
probe=$base/ni26_friday_c21gpack_live_v1_20261119
parent=$base/ni26_friday_c25p_hard_diagnosis_v1_20261119
image=$repo/runs/cloud/ni26_worker.sif
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
printf '%s  %s\n' 7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0 "$image" | sha256sum --quiet -c
verify() (
    cd "$1"
    printf '%s  source.sha256\n' "$2" | sha256sum --quiet -c
    sha256sum --quiet -c source.sha256
)
verify "$parent" 012cb9f01cb88baa0cbfdbc311bf3470c4107d5759fc35006cf926d3ae3d8ce2
verify "$f32" 2c3c4eb3ab9c8a0845a301e5f63263d6eecf6d2750eef9492980da3afa762839
verify "$owner" 3d56e3f24611b220fa017de8a49728c4ddc619eb65194c0150ee8b165bb9e436
verify "$native" 61feef26593423464c1e3d8935c0ef43ce01d737df84fa6c73035d0b3b663841
verify "$drift" 6249d01369c47e4498b5fc1cac57ca3af5ce8bb2b68dfc1165698a03a37406ba
verify "$probe" b0bc9a635389c4102107dabb9ceb7f13fc4f3a3d9f5ec2b3477160c0d267cc37
(cd "$shared" && printf '%s\n' 'f52e414494585fd2d6926e050dd2ac0462ebedc6c8f9981d16e19e6d6d02a3cb  mirror.py' | sha256sum --quiet -c)
(cd "$base" && sha256sum --quiet -c "$owner/parents.sha256")
(cd "$prior" && sha256sum --quiet -c inputs.sha256 && sha256sum --quiet -c members.sha256)
grep -Fx OK "$prior/replay-proof.txt"
grep -Fx OK "$pair/replay.log"
program=()
case "${1:?}" in
    smoke) program=(--smoke);;
    complete) test -s default_smoke.json && test -s controlled_smoke.json;;
    *) exit 2;;
esac
for mode in default controlled; do
    mkdir -p "$work/tmp_$mode" "$work/mpl_$mode"
    envs="OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/work/mpl_$mode,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/parent:/f32x:/native:/shared:/v4:/prior:/drift:/probe/friday/cap250_gate_package:/official,TMPDIR=/work/tmp_$mode"
    if [[ $mode == controlled ]]; then
        envs="$envs,CUBLAS_WORKSPACE_CONFIG=:4096:8"
    fi
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs --env "$envs" \
      --bind "$work:/work:rw,$parent:/parent:ro,$parent/diagnosis_0.json:/diagnosis.json:ro,$stock:/stock.zip:ro,$native:/native:ro,$shared:/shared:ro,$owner:/v4:ro,$prior:/prior:ro,$f32:/f32x:ro,$drift:/drift:ro,$probe:/probe:ro,$pair:/rtx_pair:ro,$other:/rtx_other:ro,$known:/known.json:ro,$rtx_known:/known_rtx.json:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
      --pwd /work "$image" python -B repeat.py "$mode" "${program[@]}"
done
