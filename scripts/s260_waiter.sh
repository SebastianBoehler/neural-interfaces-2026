#!/bin/bash
# s260 waiter: wait for the 240 pp files on Galvani, fetch, score d1 / d2 / pool, write evidence + LEDGER RESULT, commit, push.
set -uo pipefail; cd "$(dirname "$0")/.."; G=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026; P=.venv/bin/python
until [ "$(ssh -o BatchMode=yes mlcloud-galvani "ls $G/runs/causalpers/pp_*_cq50_pa2_d?.npz 2>/dev/null | wc -l")" = 240 ]; do sleep 120; done
rsync -a mlcloud-galvani:$G/runs/causalpers/pp_*_cq50_pa2_d?.npz runs/causalpers/ || exit 1
E=evidence/causalpers_14nov/s260.txt; : > $E
for d in d1 d2; do DRAW=$d $P studies/14nov/causalpers/score260.py >> $E 2>&1 || { echo FAIL $d >> $E; exit 1; }; done
$P studies/14nov/causalpers/score260.py pool >> $E 2>&1 || exit 1
V=$($P - <<'P'
import re
t=open("evidence/causalpers_14nov/s260.txt").read().split("pooled")[1]; pr=t.split("\n")
l=[x for x in pr if x.strip().startswith("k 20")][0]; lo=float(re.search(r"\[([+-][\d.]+),",l).group(1))
dr=[re.search(r"k 20 \(primary\)\s+([+-][\d.]+)",s).group(1) for s in open("evidence/causalpers_14nov/s260.txt").read().split("change vs r15")[1:3]]
ok=all(float(x)>0 for x in dr) and lo>0
print("candidate list" if ok else "undecided", "(draw k20:",dr,"pooled lower",lo,")")
P
)
{ echo "**§260 RESULT (auto-collected $(date -u '+%d %b %H:%M UTC'); evidence/causalpers_14nov/s260.txt; code db13394, Galvani cpu-galvani).** Rule verdict at k 20: $V. Prediction (+0.12 per draw, -0.02 to +0.25, draws within 0.1, >= 6/10) to be read against the table below; zero-change check passed on every order x base (scorer asserts it)."; echo '```'; cat $E; echo '```'; } >> docs/LEDGER.md
git add $E docs/LEDGER.md && git commit -q -m "s260 result: capped cpers fixed form on r15 draws 1 and 2" && git push -q
