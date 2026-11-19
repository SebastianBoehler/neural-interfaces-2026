# Forecast calibration table (19 Nov 2026)

Built from pre-registered numeric predictions with a recorded outcome: lane cards `evidence/wave_19nov/*.md`, `docs/LEDGER.md`
(F-C / D / F series, 17-19 Nov), `docs/PREDICTIONS_12NOV.md` (11-12 Nov, chain dev) and `docs/SEALED_UPLOADS.md`. Not a census of the 469
"predict" lines in the ledger: only rows with a point forecast and a measured number I could read directly. Nothing is invented; "n/g" =
no interval or number stated, "pending" = the card has no result yet. pp = balanced accuracy points. Predicted value = stated point
(else range midpoint); ratio = measured / predicted (blank when predicted 0). Instrument: S = standalone / individual-model,
F = full system (dev chain replay or native worker). Cohort: dev = ten held-new PROTEUS people (full0 / within0 = pinned order),
SCH = nine Scherer people, 8 NI-V orders or 2-4 pinned orders as stated. in/out = measured inside the stated 80 % prior or not.
Rule = the card's own refutation rule (S supported, R refuted, U undecided). Lever: NM new-person member, AD adaptation, PR personal /
returning, ST stage / routing, FM foundation model, CP capacity.

## A. Table (one row per prediction; within0 / full0 pairs share a row, each pair counts once in the statistics)

| Lane / run (date) | Lever | Instrument | Predicted pp [80 %] | Measured pp [80 %] | in/out; rule | Ratio |
|---|---|---|---|---|---|---|
| C late fusion REVE (19 Nov) | FM | F dev full0, terminal | +0.1 [-0.4,+0.6] | -0.419 [-0.977,0.000] | out (by 0.02); R | -4.2 |
| C late fusion CBraMod / both | FM | F dev full0 | 0.0 [-0.4,+0.4] / +0.1 [-0.4,+0.7] | 0.000 / -0.419 | in / out; R | n/a / -4.2 |
| C late fusion returning | FM | F P04/P10 | flat, abs < 1 | +0.400 [-0.951,+1.682] | in; U | n/a |
| C in-mixer 1/6 REVE-C / pair | FM | F dev full0 | +0.2 [-0.8,+1.2] each | +0.216 [-0.096,+0.615] / +0.147 [-0.102,+0.437] | in; U (UB < 1) | 1.08 / 0.74 |
| C in-mixer CBraMod-C | FM | F dev full0 | 0.0 [-1.0,+0.8] | +0.263 [+0.001,+0.527] | in; U | n/a |
| C repair share of cap250 misses at >=0.60 | FM | oracle | 5 % [2,12] | 27.2 % | out; n/a | 5.4 |
| C D1 pooled gain over single-corpus (C minus B) | FM | S held-new | +1.0 [-1,+4] | +0.1 (B 55.0-55.4, C 55.0-55.7) | in; U | 0.1 |
| C pooled arm entering cap250 | FM | F dev | >= +1.00 (floor) | +0.216 (REVE-C mixer) | out; R (step change) | <= 0.22 |
| C D1 smoke B minus A +5 [1,9], full-wave B >= 62 % | FM | S | see left | smoke not in card; B = 55.0-55.4 | out (62 floor); n/a | 0.89 of floor |
| C2 CBraMod minus REVE-C | FM | S held-new 8 orders | -3.0 [-7,+1] (CBra 52.0 [48,56]) | -2.46 (CBra 52.76) | in; U | 0.82 |
| C2 returning level | FM | S | 57 [50,64] | 52.30 | in | level |
| C3 Pratibha-2 (standalone 53, mixer +0.2) | FM | S / F | 53 [47,58]; +0.2 [-0.8,+1.0] | pending | | |
| D Gaussian look-ahead (19 Nov) | AD | F dev full0 / within0 | +0.15 [-0.45,+0.75] / +0.30 [-0.40,+1.00] | +0.292 [-0.060,+0.639] / +0.372 [-0.061,+0.716] | in / in; U | 1.95 / 1.24 |
| D-c class balance | AD | F dev full0 / within0 | +0.10 [-0.60,+0.80] each | +0.517 [+0.176,+0.807] / -0.205 [-0.456,-0.004] | in / in; S / R | 5.2 / -2.1 |
| D-c on Scherer 8 orders | AD | F SCH | -0.10 [-0.70,+0.50] | -0.183 [-0.427,+0.065] | in; U | 1.8 |
| D-a / D-b refit targets | AD | F | +0.20 / +0.05 | not run (stop rule, full0 +0.29 < +0.50 gate) | | |
| G g1 / g2 / g4 augmented personal members | PR | S P04+P10, 8 orders | +2.0 [-1,+5] / +0.6 [-1,+2] / +0.4 [-1,+1.8] | -4.01 / -3.66 / -1.51 (all 0/8 orders >= 0) | out x3; R | -2.0 / -6.1 / -3.8 |
| G gS (S member only; selected on same splits) | PR | S P04+P10 | +0.15 [-0.2,+0.6] | +0.66 [+0.43,+0.91] | out; selection-biased | 4.4 |
| G gS NI-V full-system (B-ret +0.1 to +0.3) | PR | F SCH | +0.2 | pending | | |
| A returning dose k=2/4/8/16/hard | PR | F dev P04/P10, 8 orders | +0.3 / +0.7 / +1.0 / +0.9 / +0.4 [about +-0.8] | no result in card (smoke 3136614 passed, array queued) | | |
| B stages B1..B5 | ST | F SCH + dev | +0.15 / +0.05 / +0.05 / 0.00 / +0.20 | pending (collector not returned) | | |
| E capacity Ea / Eb / Ec / Ed | CP | F SCH 8 orders | +0.08 / +0.10 / -0.05 / +0.15 | pending (timing and NI-V array queued) | | |
| H pooled base B-A, C-B; full cap250 | NM | S / F | +1.5 / +0.5; +0.15 [-0.5,+0.8] | pending | | |
| F-C4r balanced refit weights (18 Nov) | AD | F SCH 2 orders | +0.12 [-0.30,+0.55] | -0.291 [-0.810,+0.198] | in (edge); U | -2.4 |
| F-C8 causal arbitration | ST | F SCH 2 orders | +0.15 [-0.35,+0.65] | -0.020 [-0.205,+0.146] | in; U | -0.1 |
| F-C9 adaptive / fixed 0.75 scale | AD | F SCH | +0.30 [-0.40,+1.00] / +0.10 [-0.50,+0.70] | -0.248 [-0.665,+0.169] / +0.073 | in / in; U (adaptive minus fixed R) | -0.8 / 0.7 |
| F-C10 bounded history in raw FBCNet | AD | F SCH | +0.12 [-0.60,+0.84] | -0.255 [-0.579,+0.046] | in; U | -2.1 |
| F-C25 hard + FBL2 vs stock, external | NM | F SCH | +0.28 [-0.40,+0.96] | -0.271 [-0.757,+0.235] | in; U | -1.0 |
| F-C25 two-parent label-aware ceiling | NM | oracle SCH | +0.80 [0.25,1.55] | +1.614 [+1.349,+1.888] | out | 2.0 |
| F-C20 hard minus stock (19 Nov) | NM | F dev w0 / f0 | +0.05 / +0.05 (n/g) | +0.305 [-0.009,+0.641] / +0.331 [+0.214,+0.451] | out; U / S | 6.1 / 6.6 |
| F-C20 jitter minus stock | NM | F dev w0 / f0 | +0.25 / +0.20 | +0.193 [-0.135,+0.558] / +0.401 [+0.126,+0.652] | n/g; U / S | 0.77 / 2.0 |
| F-C21 JS, FBL2 minus stock | AD | F dev w0 / f0 | JS +0.22 / +0.15; FBL2 +0.10 / +0.10 | JS +0.350 / +0.698; FBL2 +0.383 / +0.488 (all LB > 0) | n/g; S | 1.6 / 4.7; 3.8 / 4.9 |
| F-C22 joint minus FBL2 | NM | F dev w0 / f0 | +0.24 [-0.26,+0.74] / +0.42 [-0.08,+0.92] | +0.155 / +0.290 (intervals cross 0) | in / in; U | 0.65 / 0.69 |
| F-C25d hard + FBL2 vs stock | NM | F dev w0 / f0 | +0.57 / +0.63 (n/g) | +0.574 / +0.553 | n/g; U | 1.01 / 0.88 |
| D-EAwcf cap250 minus P0r, FT+CSP stack | ST | F dev w0 | +0.40 [-0.50,+1.30] | +2.546 [+1.641,+3.406] | out; S | 6.4 |
| D-EAfcf same, full0 (re-anchored on the miss) | ST | F dev f0 | +1.4 [0.0,+2.8] | +0.667 [+0.033,+1.549] | in; S (marginal) | 0.48 |
| N12 PFT / BOTH seen (v3 chain dev) | PR | F P04+P10 | +1.4 / +2.0 | +1.07 / +1.95 | n/g | 0.76 / 0.97 |
| N12 P stage; T.C.S stage; P.T.C.S stage, all people | ST | F dev | +2.0; +3.0 [0,6]; +4.0 [1,7] | +1.96; +4.17; +4.6 | n/g; in; in | 0.98; 1.39; 1.15 |
| N12 v5.1 own stage, dJ (dev, 6 bases) | PR | F | +0.85 | +0.89 | n/g | 1.05 |
| N12 oracle gate 46a; stage under oracle gate | ST | F | +0.5 [0,+1.2]; +0.6 | -0.24; -0.43 | out; out | -0.48; -0.72 |
| N12 LADDER true labels new; chain labels new / seen | AD | F dev | +3.0 [1.8,4.5]; +0.1 [-0.4,+0.5] / +0.3 [-0.5,+1.0] | +2.0; -0.70 / -1.5 | in; out / out | 0.67; -7 / -5 |
| N12 one public session, P.T.C.S | PR | F | +1.5 [-0.5,+3.5] | +10.35 | out | 6.9 |
| N12 population: per doubling; 7 to 9 people; Mixup | NM | S core1 | +5 [3,7.5]; +1.8 [0.8,3.0]; +2 | +8.13; +3.86; -0.7 | out; out; miss | 1.63; 2.14; -0.35 |
| NI-V v1 (instrument itself) | | | no number yet; hand-over said results ~18:00 CET | none in card | | |

## B. Dev to sealed (each upload; baseline = previous sealed unless stated)

| Upload (date) | Dev forecast | Stated sealed forecast | Sealed measured | Sealed / dev | Sealed / forecast |
|---|---|---|---|---|---|
| 2 cand_rec_ts2_tsem (6 Nov) | +3.7..+3.9 J over cand_nxif | 0.8165-0.8185 | 0.8197 (+4.02); rescored +3.57 | 1.06; 0.94 | 1.06 (above range) |
| 3 route3geb (7 Nov) | +0.75 honest dev | 0.827 [0.822,0.835], +0.73 | 0.8240 (+0.43); rescored +0.55 | 0.57; 0.73 | 0.59; 0.75 |
| 4 v4.1 (11 Nov) | +0.58 | +0.2..+0.8 (mid +0.5) | +0.267 (rescored base) | 0.46 | 0.53 (in range) |
| 5 v5.1b (12 Nov) | +0.89 | 0.819 [0.817,0.824], about +0.3 | +0.361 | 0.41 | 1.2 (in range) |
| 6 cap250 (18 Nov) | public dev about +1.1; corrected full0 +0.479 [-0.222,+1.452] / matched +0.667 | none stated ("about 1:1") | +1.011 | 0.92; 2.1 / 1.5 | n/a |
| 1 cand_nxif (5 Nov) | +2.9 sel over R19e | none | 0.7795; baseline R19e never uploaded | missing | missing |
Scherer is a poor stand-in: exact cap250 vs v5.1b was +0.266 over four orders (NIV.md: "mispredicted cap250 magnitude 4x") against sealed +1.01.

## C. Summary (51 prediction pairs with a measured value; pending rows excluded; levels and ceilings excluded from ratios)

- Overall: median ratio 0.82, median bias -0.05 pp, mean bias -0.07 pp; sign right 35/51 (69 %); inside the stated 80 % interval 21/34 (62 %). The median hides a
  bimodal error: small positive forecasts landed on either side of zero (30 small full-system forecasts: mean forecast +0.22, mean measured +0.11, 18 positive, median abs error 0.29 pp).
- By lever (median ratio, n): FM 0.42 (6), AD 0.73 (17), NM 0.94 (12), PR 0.97 (9; mean bias -0.26, driven by Lane G), ST 0.98 (7). FM is the only lever with bias reliably below 1; CP has no result yet.
- Standalone vs full system: S median ratio 0.46 (n=10, mean bias -0.87 pp, sign right 6/10, 2/8 inside interval); F median 0.88 (n=41, sign 29/41, 19/26 inside).
- Cohort split inside F is the largest effect: dev held-new, positive forecasts (n=22): median ratio 1.16, mean bias +0.11, sd 0.54 pp, sign 19/22, 9/12 inside.
  Scherer and other reused external people, positive forecasts (n=6): mean forecast +0.18, mean measured -0.17, mean bias -0.35 (sd 0.19), sign right 1/6, all 6 inside their (wide) interval.
- Standalone or oracle number to full-system gain, levers with a traceable pair (12): positive 80 % lower bound on dev full system for 4 (F-C20 hard full0, FBL2, JS, CBraMod mixer at +0.001), 33 %;
  none of the 3 that were then confirmed externally held (F-C20 hard -0.063, FBL2 -0.020, F-C25 hard +0.308 then +0.050 on the exact archive). Failed: F-C13 raw S1C (+0.04), F-C32 joint
  (+0.194/+0.066), REVE (-0.42 terminal, +0.22 mixer), D Gaussian, F-C25 combined, G g1/g2/g4 (standalone already negative).
- Oracle ceiling realised as full-system gain: F-C21 27 % / 35 % (+0.35/+0.70 of +1.29/+1.98), F-C25d 40 % / 25 % (+0.57/+0.55 of +1.44/+2.25), historical §362 ceiling 14 % (+0.22 of +1.60), F-C25 external -17 %
  (-0.27 of +1.61), REVE mixer about 2 % (+0.22 of about 9.7 pp, derived: 37 % of cap250 misses at >=0.60, my arithmetic). Median 25 %; about 3 % for a weak (55 %) foreign member, about 30 % for a sibling inside the stock
  family, below 0 on external people. Repair share 27 % gave +0.2 pp.
- Dev to sealed transfer: 1.06 / 0.94 (upload 2, +3.8 step), 0.57 / 0.73 (3), 0.46 (4), 0.41 (5), 0.92 against public dev or 1.5-2.1 against corrected dev (6). Large steps about 1:1; increments below 1 pp dev
  median 0.46 (rescored 0.73 on upload 3 is the high end). Stated sealed forecasts: 2 of 2 stated ranges for uploads 4 and 5 held; upload 2 landed above and upload 3 central was 1.7x too high.

## D. Forecasting rules (derived above)

1. External or reused-people full-system forecasts: forecast 0 [-0.5, +0.3], never a positive centre. Six small positive forecasts all missed by about -0.35 pp and five of six changed sign; Scherer needs a
   separate shrink, not a smaller version of the dev number. Treat Scherer as a veto instrument, not a magnitude instrument.
2. Dev held-new full-system gains: keep the point (median ratio 1.16) but use a +-0.7 pp 80 % half-width (1.28 x sd 0.54); my stated intervals contained 62 % overall. Do not forecast a stage stack from its parts
   (D-EAwcf: 0.4 forecast, 2.5 measured, then 1.4 vs 0.67). Standalone or individual-model gains to full system: multiply by 0.1 and allow a sign flip; oracle or repair-rate ceilings: 0.03 for a weak foreign
   member, 0.25 for a stock-family sibling, 0 on external people. Treat an unselected positive standalone as a hypothesis, never as a forecast.
3. Dev to sealed: sealed = 0.5 x dev for increments below 1 pp (range 0.4-0.75), 0.9 x dev for steps of 3 pp or more; with one upload left, any candidate whose dev gain is about +0.3 and whose Scherer
   interval spans zero should be forecast at about +0.15 sealed, inside the +-0.5 dev noise, i.e. undecided. Caveat: upload 6 transferred 1.5-2.1x its corrected dev gain, so the shrink is a central rule, not a bound.
