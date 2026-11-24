#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-D3-setup
#SBATCH --output=logs/D3setup-%j.out
# release dir (ni26_laneD3_20261120; laneD3/, niv/ rsynced in; replay -> repo): returning-instrument packages ret/pkg_ret/{P04,P10}_{hardcb,D0z,D3f,D3g}, NI-V zips
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
src=$base/ni26_final_stack_ret_20261120/pkg_stack
rel=${SLURM_SUBMIT_DIR:?}; R4=$base/ni26_laneR4_20261120/ret
sif=$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif
data=$base/ni26_aux431_20261118/data/proteus_windows.npz
cd $rel; mkdir -p ret/pkg_ret ret/logs pkg_d3 zips
ln -sfn $base/neural-interfaces-2026/replay replay
cp -n $R4/dose_worker.py $R4/lane_inputs.py $R4/source.sha256 $R4/parents.sha256 ret/ 2>/dev/null || true
for who in P04 P10; do
  [ -d pkg_d3/${who}_hardcb ] || apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$base/neural-interfaces-2026/2026-competition/tracks/bci_decoding --bind "$base:$base:ro,$rel:$rel:rw" --pwd $rel \
    $sif python -B laneD3/add_d3.py $src/${who}_hardcb pkg_d3/${who}_hardcb $who $data
  [ -d ret/pkg_ret/${who}_hardcb ] || cp -rL $src/${who}_hardcb ret/pkg_ret/${who}_hardcb
  for a in D0z D3f D3g; do
    [ -d ret/pkg_ret/${who}_$a ] || python3 -B laneD3/build_drift.py $a ret/pkg_ret/${who}_$a pkg_d3/${who}_hardcb
  done
done
