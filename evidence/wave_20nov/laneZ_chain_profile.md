# Lane Z - stage-by-stage chain profile (registered Fri 20 Nov 2026, 12:55 CET, before any scoring)

Question: where does the cap250 / final-stack (hardcb) chain create accuracy, where does it erase member gains, and why. Held-new ten people, full0 and within0 (stored inputs), official cell-mean BA, equal-cell. Code `laneZ/chain/final_z.py` (copy of `laneB1/chain/final_b.py` machinery, own hooks), release dir `/mnt/lustre/work/gehler/gfh527/ni26_laneZ_20261120/chain`. Smoke 3142011 passed the zero-change guards (stock|base bitwise = release raw; FT-off bitwise = released frozen V6r path).

## Arms (every arm is one pass of the real chain; gate, FT banks, shadow and refit targets stay the frozen incumbent ones)
- prof (both systems stock=cap250, hardcb): base; bypass rebal / SessionEM / TSEM / EM+TSEM / GEB gate (g=0) / count filter / CSP; FT off (frozen V6r); all stages off (raw mean only).
- tx (both systems): synthetic gain (1-a)p + a*onehot(y) on a fixed random 10 % / 20 % of windows (`rand`) or on 10 % / 20 % of windows taken from the stock terminal's wrong windows (`wrong`), a in {0.1, 0.3, 1.0}, injected into the raw family slot (`slot`, the sc_fbraw member, weight 1 in the mixer, MASS 5/6 for hardcb) or into the whole mix (`mix`). Labels are used only for the injection.
- Jobs 3142015-3142024 (prof, tx_slot_rand, tx_slot_wrong, tx_mix_rand, tx_mix_wrong; full0 then within0).

## Pre-registered readings (numeric)
1. Stage map: raw mix BA about 73.9 (full0 stock); rebal+EM+TSEM together +1 to +3 pp over the plain weighted mean; count+CSP -0.3..+0.1 (Fable: -0.2 full). LOSO deltas (people-resampled paired 80 %): FT off -0.2..-1.0, rebal/EM/TSEM each |d| < 0.8, count and CSP each |d| < 0.4.
2. Transmission coefficient T_s = (BA gain at stage s)/(BA gain at the injection point). Mix injection: T_terminal 0.5-1.0 for random windows, below 0.5 for `wrong` windows (hard windows are where session priors and CSP pseudo-labels anchor). Slot injection: gain at the mix is about slot share (1/17 x family count) of the slot gain, so terminal gain per unit slot gain 0.03-0.08; no extra stage loss expected beyond that dilution.
3. Absorber rule: a stage is an absorber when its stage-to-stage coefficient is < 0.7 with the same sign on both orders and both systems; the loss is "dilution" when every stage coefficient is >= 0.85 and the slot-to-mix factor explains the terminal shrink. Hard windows (all 12 member slots wrong) are reported separately; injected truth cannot be sustained there only if a stage coefficient is < 0.7 on them.
4. Levers (scored in the chain on both orders, vs hardcb and vs stock; to be run after the diagnosis, details appended below before scoring): (a) raw-slot weight x2 / x4 in the mixer; (b) nested leave-one-person-out learned mixer weights (EM for mixture weights on nine people, shrink 0.5 and 1.0 toward stock weights, applied to the tenth); (c) one more form chosen from the diagnosis. Keep rule: a lever is "positive" if it beats hardcb by >= +0.2 on both orders; a gain inside +-0.5 is "undecided", not closed (AGENTS.md).

Not exercised by the replay (stated, not silent): recognition and the personal head/own-space route are unrouted for held-new people (`recognition_replayed=False`); FT labels, the gate and the shadow are frozen incumbent, so injected signal reaches them only through the live mixer, never through FT pseudo-labels.

## Lever registration (appended Fri 20 Nov, about 13:30 CET, after the prof and first tx tables, before any lever was scored)
Seen before registering: prof and tx banks of full0 and within0 (stage map, LOSO, injection). Mix injection transmits about 0.75-1.4 to the terminal; the large loss is the slot-to-mix step (a slot gain of +3.6 BA becomes +0.8 at the mix). So the levers act at the mixer only.
- `lev_w` (jobs lev_w): sc_fbraw mixer weight x2, x4, x8 (stock weight 1 of total 16), both systems. Forecast: hardcb alone slot is 59.7 vs 62-65 for the other families, so no gain: x2 within +-0.3 of hardcb, x4 and x8 <= 0 (more than -0.5 at x8).
- `lev_learn`: nested leave-one-person-out mixer weights (`laneZ/fit_weights.py`, EM likelihood weights `ll`, accuracy-tilted `acc`, shrink 50 % / 100 % toward stock weights; the held person never enters its own fit), both systems. Forecast: mix stage +0.5..+1.5 BA, terminal -0.4..+0.6 (rebalance/EM/TSEM compensate); ll100 puts weight about 0 on teacher, shallow, tf and sc_fbraw.
- `lev_x1`: pooling rule at the mixer, weighted geometric mean and a 50/50 arithmetic-geometric blend, stock weights. Forecast: terminal -1..+0.5 (rebalance and EM constants were tuned on the arithmetic mix).
Keep rule: positive if the arm beats its own system's base by >= +0.2 on both orders with the pooled paired 80 % interval above 0; within +-0.5 undecided.

## Results (scored Fri 20 Nov, ~13:45 CET; `laneZ/chain/score_z.py`, job 3142618; full tables `evidence/wave_20nov/laneZ/score_z.md|json`)
Jobs: smoke 3142011 (zero-change: stock base bitwise = release raw/filtered/terminal, FT-off bitwise = frozen V6r, hardcb base bitwise = laneB1 hardcb on all three stages, shared base arm bitwise across jobs); prof 3142015 (full0) / 3142020 (within0); tx 3142088-3142095; lev 3142321-3142324 and 3142334-3142335. Ten reused held-new people, official cell BA, paired people-resampled 80 %; dev only.

**1. Stage map (BA %, full0 / within0).**
| system | slot fbraw | mix | +rebal | +SessionEM | +TSEM | +count | +CSP = terminal |
|---|---|---|---|---|---|---|---|
| cap250 | 57.0 / 57.9 | 69.1 / 69.9 | 72.6 / 73.8 | 73.4 / 76.3 | 73.9 / 77.6 | 73.9 / 77.9 | 73.7 / 78.3 |
| hardcb | 59.7 / 58.1 | 69.2 / 70.2 | 72.7 / 73.8 | 73.3 / 76.1 | 74.1 / 77.5 | 73.9 / 77.9 | 74.3 / 78.6 |
Single members alone (cap250, full0): c1 65.6, sc_sinc 65.1, sc_fb 64.0, hb 63.6, sc_fbl 62.8, sc_rest 61.3, fbraw 57.0, shallow 53-56, tf 54-55, teacher(10) 48.6. The 13-slot weighted mean is 69.1 (+3.5 over the best member); rebalance adds +3.5/+3.9, EM +0.8/+2.5, TSEM +0.5/+1.3, count and CSP -0.2/+0.7 (full0/within0). P04 stays 49-54 at every stage (mix 49.0 -> terminal 50.0 cap250), P10 mix 68.4 -> 78.8: EM/TSEM/CSP create the new-person accuracy on people with a usable signal and nothing on P04. hardcb minus cap250 terminal +0.51 / +0.27 (mean +0.39). "Hard" windows (all 16 member slots wrong) are only 1.5 % of windows, so that split is nearly empty; the stock-raw-wrong split is in the tables.

**2. Leave-one-stage-out (terminal BA delta vs own base, mean of full0 and within0, paired 80 %):**
| bypass | cap250 | hardcb |
|---|---|---|
| rebalance | -2.40 [-3.80,-0.87] | -2.74 [-4.26,-1.06] |
| SessionEM | -0.98 [-2.16,+0.12] | -1.74 [-2.91,-0.61] |
| TSEM | -0.81 [-1.41,-0.23] | -0.98 [-1.60,-0.41] |
| EM+TSEM | -2.26 [-3.90,-0.65] | -2.51 [-4.17,-0.93] |
| GEB gate (g=0) | -0.13 [-0.60,+0.31] | -0.39 [-0.93,+0.16] |
| FT deltas | -0.37 [-1.00,+0.31] | -0.61 [-1.22,+0.03] |
| count filter | -0.30 [-0.57,-0.06] (full0 +0.04, within0 -0.63) | -0.26 [-0.46,-0.09] |
| CSP/LDA | -0.17 [-0.35,+0.03] (full0 +0.12, within0 -0.45) | -0.50 [-0.62,-0.37] |
| all stages (raw mean only) | -6.54 [-8.86,-4.02] | -6.69 [-9.01,-4.17] |
| all stages and FT | -7.96 | -8.42 |
Count/CSP are order dependent (about zero under full shuffle, +0.5 under within0). The stages are not additive (EM+TSEM together -2.3, singly -1.0 and -0.8).

**3. Transmission** (gain in BA pp at each stage relative to the injection point, pooled over the six {10,20 %} x {a=0.1,0.3,1.0} arms; per-arm values in score_z.md; cap250 / hardcb, full0 | within0):
| injected into | windows | terminal / injection point | slot -> mix | mix -> rebal | rebal -> EM | EM -> TSEM | TSEM -> count | count -> CSP |
|---|---|---|---|---|---|---|---|---|
| whole mix | random | 1.08 / 0.99 \| 0.79 / 0.76 | | 0.79 / 0.80 \| 0.78 / 0.78 | 1.08 / 1.13 | 1.17 / 1.07 \| 0.97 / 0.99 | 1.02 / 1.03 | 1.06 / 0.99 \| 0.93 / 0.89 |
| whole mix | stock-terminal wrong | 1.23 / 1.19 \| 1.18 / 1.14 | | 0.98 / 0.98 \| 1.12 / 1.12 | 1.07 / 1.09 | 1.12 / 1.11 | 1.01 | 1.04 / 0.99 \| 0.98 / 0.96 |
| raw family slot | random | 0.14 / 0.13 \| 0.19 / 0.14 | 0.17 / 0.18 \| 0.14 / 0.12 | 0.56 / 0.62 \| 0.76 / 1.12 | 1.33 / 1.49 | 1.12 / 0.69 \| 0.88 / 0.82 | 1.08 / 1.07 | 0.96 / 1.02 \| 0.95 / 0.98 |
| raw family slot | stock-terminal wrong | 0.28 / 0.24 \| 0.30 / 0.26 | 0.16 / 0.15 \| 0.10 / 0.10 | 1.01 / 0.96 \| 2.31 / 2.46 | 1.28 / 1.63 | 1.21 / 1.03 | 1.03 / 1.06 | 1.10 / 0.91 \| 0.90 / 0.82 |
Per-arm full0 hardcb: a=1.0, random 10 %: slot +3.6, mix +0.8, terminal +0.6; wrong 10 %: slot +7.2, mix +1.1, terminal +2.0.

**4. Absorber diagnosis.** Pre-registered rules applied. (i) No post-mix stage meets the absorber rule: stage-to-stage coefficients from the mix on are 0.76-1.5 and the product mix -> terminal is 0.76-1.43; rebalance removes about 20 % of a random-window gain (0.56-0.80) and EM, TSEM re-amplify (1.1-1.5), so the chain is transmission-neutral to mildly amplifying for a gain that reaches the mix. Prediction 2 (T < 0.5 for `wrong` windows) is refuted: it is above 1.0 at the mix. The count filter and CSP are neutral (0.9-1.1). (ii) The absorber is the mixer: a raw-slot gain becomes 0.10-0.18 of itself at the mix (slot weight 1 of 16 total, 6 %; the factor is about 2-3 times the weight share because small margin shifts flip close windows), so terminal / slot is 0.13-0.19 for random-window gains and 0.24-0.30 for gains on windows the system gets wrong. Forecast was 0.03-0.08, so the slot transmits about three times better than my naive weight arithmetic. (iii) Realised member gains transmit like random synthetic gains or worse: hardcb's raw slot is +2.65 BA over stock in full0 and the terminal gain is +0.51 (0.19); lane B5's +4.6 standalone, at 0.13-0.30, would predict +0.6 to +1.4 and it gave -0.4, so the loss beyond dilution is redundancy: real gains sit on windows the other members already get right (the synthetic wrong-window gains at 0.24-0.30 are what a non-redundant gain would deliver). Caveats: injections are synthetic truth on a fixed window set; FT labels, the gate and the shadow stay frozen incumbent, so the replay cannot show FT or gate feedback; recognition was not exercised (`recognition_replayed=False`); P04 is 50 % at every stage in both systems, so the chain creates nothing there and an injection moves it only where windows are flipped.

**5. Levers** (terminal BA, both orders mean, vs own base [paired 80 %]; full0 / within0 in brackets for the weight-x arms):
| lever | cap250 | hardcb |
|---|---|---|
| raw slot weight x2 | -0.15 [-0.35,+0.05] | -0.15 [-0.40,+0.11] (-0.33 / +0.04) |
| x4 | -0.89 [-1.35,-0.57] | -0.67 [-1.04,-0.38] |
| x8 | -2.59 | -2.49 |
| nested LOPO likelihood weights, shrink 50 % / 100 % | -0.24 [-0.95,+0.50] / -1.42 [-2.44,-0.41] | -0.70 [-1.23,-0.12] / -1.87 [-2.82,-0.92] |
| nested LOPO accuracy-tilted weights, 50 % / 100 % | +0.16 [-0.40,+0.71] / -0.65 | -0.16 [-0.68,+0.38] / -1.08 [-1.90,-0.28] |
| geometric pooling / 50-50 blend | -1.17 / -0.29 | -1.24 / -0.54 |
Forecasts held for w2/w4/w8 (hardcb), geometric pooling and the 100 % learned weights; the forecast "mix +0.5..+1.5 BA" for learned weights failed (the nested EM puts about 0 weight on teacher, shallow, tf and sc_fbraw and loses diversity; raw mix BA drops -0.4 to -1.9). No lever is positive on both orders (keep rule not met); none is closed beyond what it measured: ll50 and acc50 on cap250 full0 (+0.52, +0.49) are inside the +-0.5 noise and flip sign under within0 (-1.00, -0.18), so they are undecided, not a route. Stock fusion rules stay as they are.
