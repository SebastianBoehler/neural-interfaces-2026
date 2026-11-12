#!/bin/bash
# Waits for a Leonardo pack, copies its outputs home and runs the scorer through the capped runner.
# Usage: scripts/cloud/leo_collect.sh "JOBIDS" REMOTE_SUBDIR LOCAL_DIR LOG -- SCORER ARGS...   (env vars pass through)
JOBS=$1; RSUB=$2; LDIR=$3; LOG=$4; shift 5
R="ssh -o BatchMode=yes cineca-leonardo"
# JOBS: job ids, or name:<job names> for arrays (squeue -j on an array's master id stops matching once its tasks start).
# A failed or empty squeue (no connection) must not count as "finished": require a successful answer.
Q="squeue -h -j $JOBS"; case "$JOBS" in name:*) Q="squeue -h -u \$USER --name=${JOBS#name:}";; esac
while :; do OUT=$($R "$Q && echo __OK" 2>/dev/null); case "$OUT" in *__OK) [ "$OUT" = "__OK" ] && break;; esac; sleep ${INTERVAL:-1200}; done
mkdir -p "$LDIR"; rsync -az -e "ssh -o BatchMode=yes" "cineca-leonardo:/leonardo_work/AIFAC_P02_1636/ni26/$RSUB/" "$LDIR/" 2>&1 | grep -v "WARNING\|vulnerable\|upgraded"
echo "copied $(ls "$LDIR" | wc -l) files to $LDIR"; NI26_MAX_GB=24 scripts/run_capped.sh "$LOG" "$@" > /dev/null 2>&1; echo "scored -> $LOG"
