#!/bin/bash
# Source from a private, frozen Lane 5 workspace.
set -euo pipefail
export NI26_WS="$PWD"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LANE5_LIB="$NI26_WS/runs/build/liblane5.so"
export LANE5_MANIFEST="$NI26_WS/source.sha256"
: "${LANE5_MANIFEST_SHA:?pass the pinned manifest SHA256 via sbatch --export}"
PYTHON=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
check_inputs() {
    test "$(sha256sum "$LANE5_MANIFEST" | cut -d' ' -f1)" = "$LANE5_MANIFEST_SHA"
    sha256sum -c "$LANE5_MANIFEST"
    test -f "$LANE5_LIB"
}
write_gate() {
    local gate="$NI26_WS/runs/lane5_gates/$1"
    mkdir -p "$(dirname "$gate")"
    ( set -o noclobber; printf '%s\n' "$LANE5_MANIFEST_SHA" > "$gate" )
}
require_gate() {
    test "$(cat "$NI26_WS/runs/lane5_gates/$1")" = "$LANE5_MANIFEST_SHA"
}
