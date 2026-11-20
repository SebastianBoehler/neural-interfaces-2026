#!/usr/bin/env bash
# Run only within a Slurm job. Create variants outside the production package directories.
set -euo pipefail
mode=$1 src=$2 run=$3
PY=$CAP250_PYTHON
CODE=$run/src/scripts/redteam
case $mode in
  h1)
    for arm in R0z R1a R5 R2 R4; do
      for who in P04 P10; do
        "$PY" -B "$run/src/laneR4/build_keep.py" "$arm" "$run/$arm/ret_$who/pkg" "$src/hardcb/ret_$who/pkg"
      done
    done
    ;;
  dose)
    arm=${4%%:*} knob=${4#*:}
    for fold in ret_P04 ret_P10 new_P01; do
      "$PY" -B "$CODE/patch_weights.py" "$src/hardcb/$fold/pkg" "$run/$arm/$fold/pkg" --set "$knob"
    done
    ;;
  leak)
    zip=$CAP250_ASSETS/runs/sealed/final_zips/v66_cp1_cap250.zip
    printf '%s  %s\n' 9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed "$zip" | sha256sum -c -
    mkdir -p "$run/shipped"
    unzip -q -n "$zip" -d "$run/shipped"
    take=ts2_log_glob,ts2_mu,ts2_sd,ts2_W,ts2_b,v6_cf_W,v6_cf_sw,v6_cf_thr,v6_cf_lp,v6_rest_G,v6_rest_scale,em_ev_g,net34_G,net34_scale,rest_s0.pt,rest_s1.pt,rest_s2.pt,eye_s0.pt,eye_s1.pt,eye_s2.pt
    for who in P04 P10; do
      "$PY" -B "$CODE/patch_weights.py" "$src/stock/ret_$who/pkg" "$run/leak_all/ret_$who/pkg" --donor "$run/shipped" --take "$take"
    done
    ;;
  *) exit 2 ;;
esac
