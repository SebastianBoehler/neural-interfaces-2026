# MANAS shared recognition · 21 November 2026

[Preregistration](../../docs/LEDGER.md#manas-shared-recognition--21-november-registered-before-scoring) · [Earlier visit-3 audit](sota_review_21nov.md#manas-personal-head-reachability) · [Quarter diet](fable_r6_diet.md)

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
