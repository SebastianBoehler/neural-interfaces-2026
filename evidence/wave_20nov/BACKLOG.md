# Compute backlog to the deadline (Sat 21 Nov 23:59 CET)

Rule (user, 20 Nov): no cluster sits idle until the final submission. The coordinator checks Galvani, Ferranti and
Leonardo about every 50 min and starts the next unclaimed item on an idle cluster. Mark items `RUNNING <agent/lane>`
or `DONE <commit>` here. Every item pre-registers in its own card and is scored in the full system vs cap250.

Running now: final stack (Galvani; cap250 + F-C25 hard + CBraMod), lane S Shin data (Leonardo), lane V virtual
subjects (Ferranti H100).

| # | Item | Cluster | Decision value | Status |
|---|---|---|---|---|
| B1 | Final-stack arms on two more full-shuffle orders and extra seeds, plus NI-V new and B-ret, to narrow the final choice | Galvani | final upload choice | PARTIAL 20 Nov (`laneB1_final_choice.md`): seed spread ok, lite5 +0.07 / sparse20 -0.23 vs final stack (undecided, not adopted), X undecided, unrecognised timing 3,100-3,290 s; returning and zero-change jobs pending |
| B2 | Virtual subjects on the other neural families (EA FBC, FBLight, Sinc, teacher heads) if lane V standalone is positive | Ferranti / Leonardo | final upload if ready by Sat 12:00 | open (after lane V) |
| B3 | Lane V + lane S combined in one trunk | Ferranti | final upload if ready by Sat 12:00 | open (after V and S) |
| B4 | More seeds for the families whose seed variance is largest, inside the 3,600 s budget on the RTX PRO 4500 | Leonardo | small, variance only | RUNNING (lane B4, Ferranti preferred) |
| B5 | Full-system learning curve over the number of training people (5–9), to size the coverage lever | Ferranti | post-deadline direction | **DONE** 20 Nov (Galvani, not Ferranti): standalone +1.15 pp/person (7 to 9), full system -0.07 [-0.14, 0.00], negligible; laneB5_curve.md |
| B6 | Trunk pretraining on the 84-person ds007221 MI corpus (41/41 electrodes, CC0) and NETBCI, scored like lane H | Leonardo | post-deadline direction; Sat if positive | DONE (lane B6, `evidence/wave_20nov/laneB6_mi84.md`): X vs stock +0.23 [+0.10, +0.37] pp at 5/6, 1/6 -0.08; supported by its rule, one order only |
