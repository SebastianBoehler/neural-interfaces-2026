#!/bin/bash
# s251 (3): TSEM local templates on the 12 scored orders, six bases, forms tzero (plumbing) / tloc (s251 (3), TSEM local templates). Mac CPU, one process.
cd "$(dirname "$0")/../../.." || exit 1
for o in full0 full1 full2 full3 within0 within1 within2 within3 runs0 runs1 runs2 runs3; do
  BASES=B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2 .venv/bin/python studies/14nov/softfp/softoff.py $o tzero,tloc || echo "FAILED $o"
done
