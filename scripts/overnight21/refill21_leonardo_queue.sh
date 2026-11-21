#!/usr/bin/env bash
# Run from the staged ni26-afternoon-71900 directory on Leonardo.
set -euo pipefail
verify_smoke=${1:?verify smoke job}
timing_smoke=${2:?timing smoke job}
test -s jobs.tsv
if grep -q "71900 after $verify_smoke" jobs.tsv; then
    echo "Confirmation already submitted. See jobs.tsv." >&2
    exit 1
fi
submit() {
    local label=$1 id
    shift
    id=$(sbatch --parsable --kill-on-invalid-dep=yes -o "logs/%x-%A_%a.out" "$@")
    id=${id%%;*}
    printf '%s\t%s\n' "$id" "$label" | tee -a jobs.tsv >&2
    printf '%s\n' "$id"
}

# Slurm places one task and one GPU on each of three distinct nodes.
if ! grep -q timing-triplicate jobs.tsv; then
    submit timing-triplicate --job-name=ni26-719-e2e --dependency="afterok:$timing_smoke" \
    --account=AIFAC_P02_1636 --partition=boost_usr_prod --nodes=3 --ntasks=3 --ntasks-per-node=1 \
    --gres=gpu:a100:1 --cpus-per-task=4 --mem=24G --time=00:25:00 \
    --wrap='srun --label bash "$SLURM_SUBMIT_DIR/refill21_leonardo_timing.sbatch" measure' >/dev/null
fi

begin=(now 2026-11-21T13:35 2026-11-21T14:45 2026-11-21T16:25)
parents=$verify_smoke
for wave in "${!begin[@]}"; do
    ids=()
    for offset in 0 1 2 3; do
        order=$((71900 + wave * 4 + offset))
        ids+=("$(submit "$order after $verify_smoke" -J "ni26-$order" --time=00:25:00 --array=0-11%6 \
            --begin="${begin[$wave]}" --dependency="afterok:$parents" \
            verify.sbatch hard "$order")")
    done
    parents=$(IFS=:; printf '%s' "${ids[*]}")
done
submit confirmation-report -J ni26-719-report --dependency="afterok:$parents" \
    refill21_leonardo_report.sbatch >/dev/null
