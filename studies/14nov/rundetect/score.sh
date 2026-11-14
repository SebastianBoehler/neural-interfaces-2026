#!/bin/bash
# s217: the run-level count filter (runcount.py, TAU 2) with inferred run labels, one label file at a time (sequential)
cd "$(dirname "$0")/../../.."
for nm in loop40 bocpd mix loopM60 loop60 loopM40 dpoff; do
  RUN_IDS=runs/runids/$nm.npz TAU=2 .venv/bin/python studies/14nov/runcount.py within0 within1 runs0 runs1 full0 full1 > runs/rundetect/runcount_$nm.txt 2>&1
done
# control: TRUE runs passed as RUN_IDS (true12.npz, written from common.RUNI), so the filter uses the 12/12/12 cap for every
# run as it must with inferred groups; separates the cap's share of the gap from the boundaries'
RUN_IDS=runs/runids/true12.npz TAU=2 .venv/bin/python studies/14nov/runcount.py within0 within1 runs0 runs1 full0 full1 > runs/rundetect/runcount_true12.txt 2>&1
