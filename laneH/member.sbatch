#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:25:00
#SBATCH --job-name=ni26-lcmem
#SBATCH --output=member-%x-%j.out
# usage: member.sbatch TAG MASS bank_file[:bank_file...]   (lane H: absolute paths of ensemble_X_held_new.npz; MASS 0.8333333 or 0.1666667)
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
camp=$base/ni26_friday_causal_rebuild_20261118
repo=$base/neural-interfaces-2026
parent=$camp/full0_common_terminal_v1
c18=$camp/fc18_necessity_v2
c19=$camp/fc19_csp_gate_v2
c20=$camp/fc20_outer_v1
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status member.sha256
(cd "$parent" && echo '74527664de7163c98b4f633c5f926fdec471b15cc656baec7c5810b58b9d335a  source.sha256' | sha256sum --check --status && sha256sum --check --status source.sha256)
(cd "$c18" && sha256sum --check --status source.sha256)
(cd "$c19" && sha256sum --check --status source.sha256)
(cd "$base/ni26_s318_coral_fbl_20261117" && sha256sum --check --status "$parent/owner.sha256" && sha256sum --check --status "$parent/recognition.sha256" && sha256sum --check --status "$parent/complete_terminal_inputs.sha256")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$work:$parent:$c18:$c19"
tag=${1:?}; mass=${2:?}; args=()
IFS=: read -ra names <<< "${3:?}"
for n in "${names[@]}"; do
  f=$n
  args+=(--bank "$f:$(sha256sum "$f" | cut -d' ' -f1)")
done
exec "$repo/.venv/bin/python" -B full_replay.py $repo/data/proteus_windows.npz "result_$tag.json" "${args[@]}" --mass "$mass" \
  --cap250 $base/ni26_friday_c25d_held_v1_20261119/full0.npz
