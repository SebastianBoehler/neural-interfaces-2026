# D-EAr · Fresh matched returning-person EA banks

The label-blind input audit independently reproduced both historical `B2_pool` member means. It verified the 762 global rows, all eight pinned `seen_session` permutations and the four `full0` first rows. Each fresh fit includes the held person's first visit and excludes their later two visits. P04 source/target counts are 2,278/377. P10 source/target counts are 2,270/385. The archive checks use prior predictions as references. The new weights come from fresh matched fits. [Input receipt](inputs-audit-18nov.json) · [Returning input contract](../causal_replay/REBUILD.md#returning-person-inputs)

The wrapped actors use c625ecc7 and the previously verified deterministic replay helper. The two P10/seed 0 smoke fits reproduced their fitted tensors and held single-window probabilities exactly. Shallow smoke 3131025_11 completed in 1m04s. The first FBC smoke's verification treated inactive denominator parameters as if the FFT forward consumed them. Corrected FBC smoke 3131070_1 completed in 2m39s and confirmed finite, nonzero gradients and updates in all nine FIR numerator tensors. Minimum per-tensor maximum coefficient change was 0.02781. All nine inactive denominators had zero change and no gradient.

The v2 source manifest is `e26cd404c172b0c7734029cfb0a57191fecbfef691aa227d23c3e479057e27fa`. The original successful Shallow receipt `7a1f866744ba7fe58eb7b7ddf77e5cc4ea0ec77b9f06c61dba3ac75d3f67d4d1` remains under its original manifest and is allowed only for that bank position. FBC retry receipt `235f50337da76b58760767d61c8e7ac90316509c3c36e1784a975d72bf3f554c`. Sealed paired smoke manifest `97b6e2428697f6bf45602d912ccf2bd5730fd3071c3bafa2c7cbbf09a62754a6`.

The remaining twenty Galvani fits 3131131 finished with exit 0, at concurrency six. Collector 3131132 completed at 20:53:03 CET on 18 November. Slurm accepted the successful FBC array dependency. The production hashes independently require the completed Shallow smoke. The fits read only source labels. The collector reads metadata and predictions and calculates no held accuracy.

## Authenticated banks

Both families use the same sorted 762 global rows and eight pinned deliveries in `/mnt/lustre/work/gehler/gfh527/ni26_friday_ea_returning_fresh_v2_20261118/`. Each NPZ includes `index`, `seeds`, `orders`, `delivery`, `proba` and the matching `recording` probabilities.

| Bank | Slot | Causal array shape | SHA-256 |
|---|---:|---:|---|
| `fbcnet_ea_tf_seen_session_orders.npz` | 1 | 5 × 8 × 762 × 3 | `34bc829e8fcde85a51eaefae4788c886e06e44d648f4d7f3fb7c104ca6b229e3` |
| `shallow_ea_seen_session_orders.npz` | 3 | 6 × 8 × 762 × 3 | `6809e79d48910ce3ccf07ef5244d8b19ef7c66c2cdbe5f4645d06ebbba5d7a6b` |

All 22 fits passed their causal prefix, interleaved reset and first-window EA checks. The largest checked single-window versus batch deviations were 0.0000011293 for FBCNet and zero for Shallow, with exact decisions in both. All ten FBCNet fits used and updated nine FIR numerator tensors each, and their inactive denominators stayed fixed. Across the two new banks, causal delivery changed probabilities from the matching fresh recording predictions by maxima of 0.59091 and 0.98045.

The fresh FBCNet recordings matched all 3,810 archived argmax decisions and differed by at most 0.00000034496 in probability. The fresh Shallow recordings changed 62 of 4,572 archived argmax decisions and differed by up to 0.38949. This describes numerical agreement between source-matched model executions. The two comparison branches can use identical new banks. The archived input bank provides a separate historical zero control.

Full Galvani `returning_result.json` SHA-256 `bc3fed6fc7f23b61b6627159d428d30ff9837fcd779c6970116bda4c474555aa`. The receipt includes individual fit hashes and checks. [Input contract and scope](../causal_replay/REBUILD.md#returning-person-inputs)
