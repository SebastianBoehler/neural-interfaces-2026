# MANAS shared recognition · 21 November 2026

[Preregistration](../../docs/LEDGER.md#manas-shared-recognition--21-november-registered-before-scoring) · [Earlier visit-3 audit](sota_review_21nov.md#manas-personal-head-reachability) · [Quarter diet](fable_r6_diet.md)

**Keep the signed quarter diet.** The MAIN override gained −0.053 pp at 50/50 on held orders, with a paired 80% interval of [−0.286, +0.185] pp.

## Personal-head activation

The signed quarter ZIP is `42fb63cf1ae9f54cf48315b00894b3788bc65400e75e06bd208ffe3753c43b07`. Its MANAS A1 bank uses visit 1 for both the gallery and personal classifiers. Each fingerprint combines the 820 off-diagonal channel correlations with 246 band-power features. A1 compares the running prefix mean to the gallery by root-mean-square difference, scaling each coordinate by `1 / max(gallery_std, 0.08)`. The radius is 0.8 times the 10th percentile of inter-person gallery distances (signed value `0.5788459182`). Every fourth delivery from 12, the closest candidate must be within that radius and its distance divided by the runner-up distance must be below 0.80. The same enrolled candidate must pass two consecutive checks. A1 then mixes 70% personal logits with 30% adapted generic logits. The full stack applies a 10% MANAS posterior weight after CSP/count. [Online gate](../../cleanslate/teamA/online.py) · [Fingerprint](../../cleanslate/teamA/eeg.py) · [Gallery fit](../../cleanslate/teamA/gallery.py)

| Streams | Fit 71800–71807 | Held 71808–71815 | All orders |
| --- | ---: | ---: | ---: |
| Returning visit 2 | 7/16 | 6/16 | 13/32 |
| Returning visit 3 | 0/16 | 0/16 | 0/32 |
| Held-new folds | 1/112 | 4/112 | 5/224 |

All visit-2 activations were P10, at deliveries 16–148. For P04 visit 2, P07 or P08 ranked first at every checkpoint and even the best distance was 0.624. P04 visit 3 occasionally passed the radius, but its smallest ratio was 0.826. P10 visit 3 never passed the radius or ratio, with minima 0.858 and 0.841. All held-new activations were P04 visit 3 routed to P07 with P04 excluded from the gallery. MAIN latched on all 64 returning streams and one fit held-new stream (P10 visit 1 to P03, order 71802). [Gate replay](../../scripts/bottleneck_21nov/manas_head_gate.py) · [MAIN recognition receipts](fable_r6_diet.md#fit-orders-7180071807)

The earlier Bottleneck capture stored CSP, MANAS, and MAIN recognition through 27,336 deliveries across 144 pinned streams. Its terminal output agrees with the 90/10 CSP/MANAS blend to `1.70e-8`. The saved MANAS tap contains the final posterior. The gate replay and new live tap recover the actual head identity. [Bottleneck capture](bottleneck_review_21nov.md)

Each of the 16 fit settings had an impostor (range 1–69/112). The native setting already admitted P04/3 as P07 at age 24 on order 71805. Increasing either bound preserves those accepted checkpoints or admits an earlier impostor in the same fold. The relaxation therefore had no eligible fit setting and did not advance to decoder scoring.

Leonardo Slurm 65046913 verified the signed production archive and all 12 MANAS fold weight digests against the pinned fixture receipts. The reconstructed galleries reproduced every exact-bank activation, identity and delivery. Fit receipt SHA-256 `9414df9287a7bd24e7a081d883739dfc9b58f50e0ced2945bd8ab89a909af91b`. Held receipt SHA-256 `30e18a0ad76b7cf6da0e7c0b204d85db2bc1f23531a253cee7197e61531291c9`.

## MAIN latch override

The override retains MANAS's native A1 readout until MAIN's first positive recognition ramp. On that delivery it maps MAIN's winning person name to the sorted MANAS enrollment bank and uses that head for the rest of the stream. The alternate readout uses the current captured feature vector and the same updated generic logits. The adapter continues to update from the same features. Rebalance, both quarter-weight EM stages, count bypass, CSP and the terminal 10% blend follow their signed paths. Each replay verifies native labels and quarter-diet labels against the pinned captures and checks route-score equality and causal prefixes. [Patch](../../scripts/bottleneck_21nov/manas_head_patch.py) · [Replay worker](../../scripts/bottleneck_21nov/manas_head_worker.py)

The recording-order v3 check printed `OK` locally and on Leonardo. Smoke 65046823 reproduced 189 pinned P04/2 labels on order 71800. MAIN latched as P04 at delivery 41. Both live Solver sources matched their 84-window tape prefixes exactly in labels and within `2e-5` in RouteB and terminal scores. The alternate posterior changed after the latch. Smoke receipt SHA-256 `13f586b9a200a9cebc28d044441196edcde9d8fd8922e757d60adea2b495feb4`. Source revision `59125266`.

Fit array 65046825 completed all five slices. At 50/50 the signed quarter control scored 78.032% BA. The override scored 78.016%, a delta of −0.016 pp [paired 80% −0.149, +0.151]. Held-new was +0.000 pp and returning was −0.032 pp [−0.298, +0.302]. P04 gained +0.137 pp and P10 lost −0.201 pp. The two new decision changes both came from the MAIN misrecognition of P10/1 as P03 on order 71802. There were 81 returning decision changes. Fit receipt SHA-256 `be794b5631883d811acc62859b3727455c07d6c4be9d995ac28780ea6c1f6206`. Held array 65047224 was released after fit validation at 18:17:55 CET.

### Held orders 71808–71815

All four Leonardo slices in Slurm array 65047224 completed. The 80% intervals use 20,000 paired draws of people and orders, with the official balanced cell weights. Each named returning person's interval resamples orders. The fit and held blocks use the same ten new and two returning people.

| Cohort | Quarter BA | MAIN override BA | Δ pp [80%] | Changed labels |
| --- | ---: | ---: | ---: | ---: |
| Held-new | 74.397% | 74.397% | +0.000 [0.000, 0.000] | 0/21,240 |
| Returning | 81.981% | 81.874% | −0.107 [−0.573, +0.370] | 91/6,096 |
| 50/50 | 78.189% | 78.135% | −0.053 [−0.286, +0.185] | — |

P04 gained +0.331 pp [+0.102, +0.556], and P10 lost −0.545 pp [−0.674, −0.416]. The held MAIN recognition receipts contain zero impostors. All held-new streams therefore use the original emissions, including the four native MANAS false matches. The 50/50 point and lower bound miss both promotion thresholds. The results apply to this two-person returning instrument. Exact-archive qualification was conditional on promotion.

Held receipt SHA-256 `86b31f9f07aeffd952e9b3ade9e0194efc0faa0451ff5fd02aecb950390ab0d5`. The collector verified the fold, order, activation, identity and control labels against the original tapes before scoring. [Collector](../../scripts/bottleneck_21nov/manas_head_report.py)
