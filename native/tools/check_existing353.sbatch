#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-native353-existing
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
task_base=/mnt/lustre/work/gehler/gfh527
task_source="$task_base/ni26_native353_full_v1_20261117"
task_repo="$task_base/neural-interfaces-2026"
task_assets="$task_base/ni26_native353_qualify_v1_20261117/package"
task_python="$task_repo/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export LD_LIBRARY_PATH="$task_base/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sha256sum -c source.sha256 >/dev/null
test_status=0
run_check() {
 local name="$1"
 shift
 if "$@" > "$name.log" 2>&1; then echo "PASS $name"; else echo "FAIL $name"; test_status=1; fi
}
run_check components "$task_source/build/ni26_probe" "$task_assets/native.pt" "$task_assets/source_fixture.pt" component_output.pt
if test -r component_output.pt; then run_check component_parity "$task_python" "$task_source/tests/check_probe.py" "$task_assets/source_fixture.pt" component_output.pt; fi
run_check routing "$task_source/build/ni26_routing_source" "$task_assets/native.pt" "$task_base/ni26_native353_routing_components1/reference/fixture.pt"
run_check linear_fixture "$task_python" "$task_source/tests/linear_fixture.py" linear_fixture.pt
if test -r linear_fixture.pt; then run_check linear "$task_source/build/ni26_linear_source" linear_fixture.pt linear_output.pt; fi
exit "$test_status"
