#!/bin/bash
# Waits for a Leonardo pack, copies its outputs home and runs the scorer through the capped runner.
# Usage: scripts/cloud/leo_collect.sh "JOBIDS" REMOTE_SUBDIR LOCAL_DIR LOG -- SCORER ARGS...   (env vars pass through)
JOBS=$1; RSUB=$2; LDIR=$3; LOG=$4; shift 5
H=${HOST:-cineca-leonardo}; ROOT=${RROOT:-/leonardo_work/AIFAC_P02_1636/ni26}  # HOST=mlcloud-galvani RROOT=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026 for Galvani
R="ssh -o BatchMode=yes $H"
# JOBS: job ids, or name:<job names> for arrays (squeue -j on an array's master id stops matching once its tasks start).
# A failed or empty squeue (no connection) must not count as "finished": require a successful answer.
# squeue -j errors ("Invalid job id") once a job has left the queue, so ids are matched in the user's own list instead
Q="squeue -h -u \$USER -o %i | grep -E '^(${JOBS//,/|})(_|\$)' ; true"; case "$JOBS" in name:*) Q="squeue -h -u \$USER --name=${JOBS#name:}";; esac
while :; do OUT=$($R "$Q && echo __OK" 2>/dev/null); case "$OUT" in *__OK) [ "$OUT" = "__OK" ] && break;; esac; sleep ${INTERVAL:-1200}; done
# a failed job must be as loud as a finished one: print every job's final state
ST=$($R "sacct -X -n -P -u \$USER -S now-3days -o JobID,JobName,State,ExitCode | grep -E '$(J=${JOBS#name:}; echo ${J//,/|})' | tail -n 60" 2>/dev/null); echo "$ST"
echo "$ST" | grep -q -E "FAILED|CANCELLED|TIMEOUT|OUT_OF_ME|NODE_FAIL" && echo "WARNING: at least one job did not complete"
mkdir -p "$LDIR"; rsync -az -e "ssh -o BatchMode=yes" "$H:$ROOT/$RSUB/" "$LDIR/" 2>&1 | grep -v "WARNING\|vulnerable\|upgraded"
echo "copied $(ls "$LDIR" | wc -l) files to $LDIR"; NI26_MAX_GB=24 scripts/run_capped.sh "$LOG" "$@" > /dev/null 2>&1; echo "scored -> $LOG"
