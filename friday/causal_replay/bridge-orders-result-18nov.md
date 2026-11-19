# Held-new EA bridge across all pinned orders

All seven additional orders passed and have separate read-only historical and corrected inputs for matched `r15c250` refits. Each order reproduced the archived P0r posteriors and rebalancing and gate traces exactly on 2,655 rows. Both replacement slots reached the live mixer. Each replay produced 2,515 finite Bio scores and routed zero of 14 sessions in both arms.

The independent prefix and reset checks matched exactly. Each order rebuilt 240 refit states and 24,180 strict-past pseudo-label vectors per arm. Current EM/TSEM emissions matched within 3.34e−16 across the matrix. An independent read verified each alias hash, pinned delivery permutation, canonical shadow tensor and strict-past mask.

| Order | Maximum corrected shadow change | Legacy vs regenerated historical strict-past decisions changed | Maximum historical probability difference |
| --- | ---: | ---: | ---: |
| within0 | 0.067949 | 28 | 0.159759 |
| within1 | 0.078479 | 25 | 0.094125 |
| within2 | 0.081605 | 42 | 0.114333 |
| within3 | 0.061893 | 18 | 0.125474 |
| full1 | 0.172533 | 18 | 0.079033 |
| runs0 | 0.100716 | 0 | 0.00009566 |
| runs1 | 0.114242 | 12 | 0.162384 |

The initial within0 smoke reached the refit state but stopped at the guard that forbids loading legacy pseudo-labels during the live replay. The retry moved that comparison after the guarded replay. Galvani 3131072 passed within0 at 20:45:33 CET. Array 3131078 completed the remaining six orders by 20:46:51 CET.

Read-only release `/mnt/lustre/work/gehler/gfh527/ni26_friday_causal_rebuild_20261118/orders_v2`. Its `orders_summary.json` records every order's status, both alias paths and hashes, and the separate legacy comparison. Manifest SHA-256 `c3015d803a90061a57a211633adbefd20ed970f6687c0938316c615e831e7c73`. Source manifest SHA-256 `43cbefd904c809898e551cffced378f13961f971ba04389cb0dd6c9f9b9d3ea0`.

Use each arm's own `chain_v51b_<order>.npz` for both `CHAIN_NPZ` and `RELAB_NPZ`. The [existing full0 release](bridge-result-18nov.md) supplies the eighth order. These inputs qualify matched refits. Full-system scoring remains gated on the matched refits and downstream causal replay.
