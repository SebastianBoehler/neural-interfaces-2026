#!/bin/bash
# leo_collect.sh line 10-11, with ssh replaced by a local shell and squeue by a command that fails (as a Slurm controller timeout would).
# The loop's exit condition is OUT == "__OK". Prints what the waiting loop would see.
JOBS=64432182
Q="squeue_that_fails -h -u \$USER -o %i | grep -E '^(${JOBS//,/|})(_|\$)' ; true"
OUT=$(bash -c "squeue_that_fails() { echo 'slurm_load_jobs error: Socket timed out' >&2; return 1; }; $Q && echo __OK" 2>/dev/null)
echo "job-id form, squeue fails:   OUT='$OUT' -> $([ "$OUT" = "__OK" ] && echo 'loop ends: jobs counted as finished' || echo 'keeps waiting')"
Q="squeue_that_fails -h -u \$USER --name=x"
OUT=$(bash -c "squeue_that_fails() { echo 'slurm_load_jobs error' >&2; return 1; }; $Q && echo __OK" 2>/dev/null)
echo "name: form, squeue fails:    OUT='$OUT' -> $([ "$OUT" = "__OK" ] && echo 'loop ends: jobs counted as finished' || echo 'keeps waiting')"
