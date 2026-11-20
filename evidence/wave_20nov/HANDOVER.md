# Handover · Fri 20 Nov ~12:30 CET (Mac going offline)

Cluster jobs keep running; local agents and the hourly keeper stop while the Mac is offline. On return: re-check
ssh (Galvani 2FA, Leonardo `step ssh login`; cert expires 22:39), read the results below, restart agents as needed.

## Results that land unattended (collectors chained with afterok)
- Lane P (stack combinations, Galvani `ni26_laneP_20261120`): full0 3141462, within0 3141463 → score 3141464. Card `laneP_stack.md`.
- Lane N (finalists on NI-V, 8 orders, Galvani `ni26_niv_v1_20261119/out`): summary 3141005 → `out/laneN_summary.{json,md}`. Card `laneN_niv_final.md`.
- Lane O (8-order proxy, Galvani `ni26_laneO_20261120`): smoke 3140866 → array 3140891 → scorer 3140892 → `score_o.{json,md}`. Card `laneO_orders.md`.
- Lane B1 leftovers: zero-change 3140617/3140618; sparse20 returning 3140818 → 3140819. Card `laneB1_final_choice.md`.

## Needs an agent to continue (stops while offline)
- cap250 fold pipeline (Codex lead; SPEC `cleanslate/cap250/SPEC.md`, card `cap250_pipeline.md`): families built except SessionBN
  (builder still running); parity chain on Leonardo 64932077 → 64932082 → 64932085. Next: assemble, fresh all-ten parity, launch folds.
- Team C (clean slate, `cleanslate_teamC.md`): ~6 pp below cap250; not an upload candidate.

## Decision state
Upload candidate: final stack hardcb (`runs/sealed/final_zips/final_stack_hardcb.zip`, sha 958fd3f3…), +0.39 pp held-new dev,
~+0.2 sealed, ~3,130–3,150 s projected. Replace only if lane P/N/O show a better joint arm that fits 3,600 s.
Upload Sat ~11:00–14:00 with the user's explicit go-ahead; ask organisers whether a timeout counts as Failed.
