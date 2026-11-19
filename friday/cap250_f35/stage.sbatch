#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani,2080-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-f35d
#SBATCH --output=stage-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
camp=$base/ni26_friday_causal_rebuild_20261118
work=$base/ni26_friday_f35c_held_20261119
release=${SLURM_SUBMIT_DIR:?}
cd "$release"
sha256sum --check --status stage.sha256
(
    cd "$work"
    printf '%s\n' \
        'cecca57dcebb3b7dae4c6e84b9380543d59bd86a511c178adb5b15478fe37c9d  whole.sha256' \
        '99614eeeac57e914d14f3c4ccdd162a14ea6f9e398b99dbc4194b263b59813c7  pupil.sha256' |
        sha256sum --check --status
    sha256sum --check --status whole.sha256 pupil.sha256
)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$work:$camp/f25c_v1:$camp/fc20_outer_v1:$camp/within0_terminal_shared_v1:$camp/fc18_necessity_v2:$camp/fc19_csp_gate_v2"
if [[ $# -eq 0 ]]; then
    test -s smoke.json
fi
exec "$base/neural-interfaces-2026/.venv/bin/python" -B stage_score.py "$release" "$@"
