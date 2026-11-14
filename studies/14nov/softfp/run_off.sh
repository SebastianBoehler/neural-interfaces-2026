#!/bin/bash
# s251 (2): soft offsets on the 12 scored orders, six bases, forms zero (plumbing check) / kern / top10. Mac CPU, one process.
cd "$(dirname "$0")/../../.." || exit 1
for o in full0 full1 full2 full3 within0 within1 within2 within3 runs0 runs1 runs2 runs3; do
  BASES=B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2 .venv/bin/python studies/14nov/softfp/softoff.py $o zero,kern,top10 || echo "FAILED $o"
done
