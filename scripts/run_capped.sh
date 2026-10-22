#!/usr/bin/env bash
# Run one local job with hard memory limits (the Mac shares RAM with the GPU).
# Usage: scripts/run_capped.sh LOG CMD [ARGS...]
#   NI26_MAX_GB (default 8): kill the job when its process tree exceeds this RSS.
#   NI26_MPS_RATIO (default 0.25): PyTorch MPS high-watermark ratio; the MPS
#     allocator raises an out-of-memory error instead of growing into swap.
# Jobs queue on a lock, so at most one capped job runs at a time.
set -uo pipefail
log=$1; shift
max_kb=$(( ${NI26_MAX_GB:-8} * 1024 * 1024 ))
lock="${TMPDIR:-/tmp}/ni26_run_capped.lock"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${NI26_MPS_RATIO:-0.25}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=$(awk "BEGIN {print 0.8 * $PYTORCH_MPS_HIGH_WATERMARK_RATIO}")
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}

tree_rss() {  # RSS (kB) of a process and all its descendants
  local pids=$1 all=$1 next
  while [ -n "$pids" ]; do
    next=$(pgrep -P "$(echo "$pids" | tr ' ' ',')" 2>/dev/null | tr '\n' ' ')
    all="$all $next"; pids=$next
  done
  ps -o rss= -p "$(echo $all | tr ' ' ',')" 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

until mkdir "$lock" 2>/dev/null; do sleep 20; done  # one job at a time
trap 'rmdir "$lock" 2>/dev/null' EXIT
echo "[run_capped] $(date) start: $* (max ${NI26_MAX_GB:-8} GB)" >> "$log"
"$@" >> "$log" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  rss=$(tree_rss "$pid")
  if [ "$rss" -gt "$max_kb" ]; then
    echo "[run_capped] $(date) killed: RSS $((rss / 1024)) MB > limit" >> "$log"
    pkill -TERM -P "$pid"; kill -TERM "$pid"; sleep 5; kill -KILL "$pid" 2>/dev/null
    exit 137
  fi
  sleep 10
done
wait "$pid"; rc=$?
echo "[run_capped] $(date) exit $rc" >> "$log"
exit $rc
