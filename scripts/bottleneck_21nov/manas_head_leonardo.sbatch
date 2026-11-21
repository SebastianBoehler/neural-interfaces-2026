#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --cpus-per-task=2
#SBATCH --time=00:16:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
all="$WORK/ni26/cap250/runs"
root="$all/leonardo-fill-ba-six-21nov"
source "$root/env.sh"
launcher="$CAP250_RELEASE/cleanslate/cap250/slurm.sh"
export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
here="$run/src/scripts/bottleneck_21nov"
diet="$all/fable-r6-diet-21nov"
archive="$all/bottleneck-returning-em-21nov/out"
phase=${1:?}
case "$phase" in
    gate)
        for split in fit held; do
            bash "$launcher" "$here/manas_head_gate.py" "$split" "$run/in/quarter.zip" \
                "$CAP250_DATA" "$root/grow/held" "$run/gate" --fixtures "$root/grow/fixtures"
        done
        ;;
    smoke)
        bash "$launcher" "$WORK/ni26/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
        grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
        exec bash "$launcher" "$here/manas_head_worker.py" "$CAP250_DATA" "$root/grow" \
            "$archive/fit" "$diet/out" "$diet/em_archive_manifest.json" "$run/smoke" 22 --smoke
        ;;
    fit|held)
        exec bash "$launcher" "$here/manas_head_worker.py" "$CAP250_DATA" "$root/grow" \
            "$archive/$phase" "$diet/out" "$diet/em_archive_manifest.json" "$run/out/$phase" \
            "${SLURM_ARRAY_TASK_ID:?}"
        ;;
    *) exit 2 ;;
esac
