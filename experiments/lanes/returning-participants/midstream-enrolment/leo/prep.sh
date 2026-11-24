#!/bin/bash
# Leonardo release dir: patched copies of the NI-V worker/enrolment (container paths -> this cluster). Run as a batch step, not on the login node.
set -euo pipefail
cd "${R4:?}"
mkdir -p niv_leo
for f in niv/*.py; do
  sed -e "s#/official/tracks/bci_decoding#$OFF/tracks/bci_decoding#g" -e "s#\"/external.npz\"#\"$EXT\"#g" -e "s#\"/rows.tsv.gz\"#\"$R4/scherer_rows.tsv.gz\"#g" "$f" > niv_leo/$(basename $f)
done
grep -c "$OFF" niv_leo/worker.py niv_leo/build_enrolled.py
mv -n ni26_niv_v1_20261119/zips/* zips/ 2>/dev/null || true
mv -n ni26_laneR4_20261120/ret/zips/* zips/ 2>/dev/null || true
rm -rf ni26_niv_v1_20261119 ni26_laneR4_20261120
ls zips
