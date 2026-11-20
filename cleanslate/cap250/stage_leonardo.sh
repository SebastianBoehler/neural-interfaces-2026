#!/usr/bin/env bash
# Stage one committed revision for the offline Leonardo workers.
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
rev=$(git rev-parse "${1:-HEAD}^{commit}")
official=$(git rev-parse "$rev:2026-competition")
host=cineca-leonardo
base=$(ssh -o BatchMode=yes "$host" 'printf "%s\n" "$WORK/ni26/cap250"')
if [[ "$base" != /leonardo_work/*/ni26/cap250 ]]; then
  printf 'Unexpected Leonardo WORK path: %s\n' "$base" >&2
  exit 1
fi
release=$base/releases/$rev
pending=$(ssh -o BatchMode=yes "$host" "mkdir -p '$base/releases' &&
  test ! -e '$release' && mktemp -d '$base/releases/.stage-$rev.XXXXXX'")
git archive "$rev" cleanslate/cap250 cleanslate/harness experiments scripts lean \
  submissions/lean_v66 submissions/final_stack friday/cap250_hard_package \
  friday/cap250_raw_swap/model laneT laneC laneC2 studies/11nov/rest4 studies/12nov |
  ssh -o BatchMode=yes "$host" "tar xf - -C '$pending'"
git -C 2026-competition archive "$official" benchmark_utils tracks/bci_decoding |
  ssh -o BatchMode=yes "$host" "mkdir '$pending/2026-competition' && tar xf - -C '$pending/2026-competition'"
ssh -o BatchMode=yes "$host" "mkdir '$pending/data' &&
  ln '$base/assets/data/proteus_windows_start-4.5.npz' '$pending/data/proteus_windows_start-4.5.npz' &&
  test ! -e '$release' && mv -T '$pending' '$release'"
printf '%s\n' "$release"
