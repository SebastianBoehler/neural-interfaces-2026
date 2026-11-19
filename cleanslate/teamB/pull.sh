#!/bin/bash
# Fetch a team B dev run from Galvani and score it locally: ./pull.sh TAG [grep pattern for collect lines]
tag=$1; pat=${2:-@base}
here=$(cd "$(dirname "$0")/../.." && pwd); dst=$HOME/ni26_csB/$tag; mkdir -p "$dst"
for i in 1 2 3 4 5 6 7 8; do rsync -e "ssh -o BatchMode=yes" -a --include 'dev_*.npz' --exclude '*' \
  mlcloud-galvani:/mnt/lustre/work/gehler/gfh527/ni26_csB_20261120/$tag/run/ "$dst/" 2>/dev/null && break; sleep 8; done
export CS_DATA=$here/data/proteus_windows.npz CS_OFFICIAL=$here/2026-competition/tracks/bci_decoding
echo "## $tag: $(ls "$dst" | grep -c dev_) folds"
"$here/.venv/bin/python" "$here/cleanslate/teamB/collect.py" "$dst" ${REF:+--ref "$REF"} | grep -E -- "$pat"
"$here/.venv/bin/python" "$here/cleanslate/teamB/diag.py" "$dst"
