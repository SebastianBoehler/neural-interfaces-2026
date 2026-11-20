# Compute backlog to the deadline (Sat 21 Nov 23:59 CET)

Rule (user, 20 Nov): no cluster sits idle until the final submission. The coordinator checks Galvani, Ferranti and
Leonardo about every 50 min and starts the next unclaimed item on an idle cluster. Mark items `RUNNING <agent/lane>`
or `DONE <commit>` here. Every item pre-registers in its own card and is scored in the full system vs cap250.

Running now: final stack (Galvani; cap250 + F-C25 hard + CBraMod), lane S Shin data (Leonardo), lane V virtual
subjects (Ferranti H100).

| # | Item | Cluster | Decision value | Status |
|---|---|---|---|---|
| B1 | Final-stack arms on two more full-shuffle orders and extra seeds, plus NI-V new and B-ret, to narrow the final choice | Galvani | final upload choice | PARTIAL 20 Nov (`laneB1_final_choice.md`): seed spread ok, lite5 +0.07 / sparse20 -0.23 vs final stack (undecided, not adopted), X undecided, unrecognised timing 3,100-3,290 s; returning and zero-change jobs pending |
| B2 | Virtual subjects on the other neural families | — | — | CLOSED: lane V refuted (spatial transforms −2 to −17 pp) |
| B3 | Lane V + lane S combined in one trunk | — | — | CLOSED: V and S both refuted |
| B4 | More seeds for the families whose seed variance is largest, inside the 3,600 s budget | Leonardo | small, variance only | open (agent died in the 20 Nov outage; restart if a cluster is idle) |
| B5 | Full-system learning curve over the number of training people (5–9), to size the coverage lever | Ferranti | post-deadline direction | **DONE** 20 Nov (Galvani, not Ferranti): standalone +1.15 pp/person (7 to 9), full system -0.07 [-0.14, 0.00], negligible; laneB5_curve.md |
| B6 | Trunk pretraining on the 84-person ds007221 MI corpus (41/41 electrodes, CC0) and NETBCI, scored like lane H | Leonardo | post-deadline direction; Sat if positive | DONE (lane B6, `evidence/wave_20nov/laneB6_mi84.md`): X vs stock +0.23 [+0.10, +0.37] pp at 5/6, 1/6 -0.08; supported by its rule, one order only |

## Overnight items (added Fri 20 Nov 13:10 CET; running lanes: P2 packaging, Z stage profile, R1 big stack, R4 keep-adapt, cap250 fold pipeline)

| # | Item | Cluster | Decision value | Status |
|---|---|---|---|---|
| B7 | Second CBraMod/hard seed set and lite5 seeds for lane P's best joint arm (ds_all / ds_l5) in the chain, both orders: is its +0.86 seed-robust? | Galvani | final upload | DONE Fri 14:25 CET: ds_all seed-robust vs cap250 (13/13 sets positive, pooled +0.72 [+0.11,+1.26]), vs final stack +0.33..+0.38 undecided; laneB7_seeds.md |
| B8 | Once the cap250 fold waves run: add ds_all (and R4/R1 winners if any) as extra fold variants in the same DAG | Leonardo | real-package check of the upload | open (after fold waves start) |
| B9 | NI-V B-new and B-ret on 8 orders for every new packaged candidate from P2, R4, R1 (Scherer-free CBraMod) | Galvani | external veto for the upload | open (after each package exists) |
| B10 | Returning-person stress set: simulate one-session people from P04/P10 sessions and from NETBCI cross-day data, score stock vs R4 arms | Leonardo | sizes the 43 % returning lever | open |
| B11 | Paired A/B/B/A runtime for every new candidate zip on 2080 Ti and A100, unrecognised and recognised sessions, sealed projection | Galvani | upload safety | open (after each package exists) |
