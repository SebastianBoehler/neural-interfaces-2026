# Fresh Shallow EA bank · D-EAn

All 60 fresh source-excluded fits completed on Galvani A100. The six seeds and ten original all-LOSO folds supplied 127,440 causal probability vectors across eight pinned delivery orders and 15,930 recording-order controls. Each pair used the same fitted model, and both full-system comparators must read this identical bank.

All 756 session/order checks reproduced the direct first input, first 16 predictions and interleaved resets bit for bit. Maximum posterior discrepancy against the producer's batched prefix was exactly zero. P10 seed 0 repeated its fit and all nine paired replays bit for bit across three sessions. The collector verified every per-fit receipt, checkpoint and probability file, distinct trained states, and matching source labels and alignment priors across seeds within each fold. An independent read confirmed the final bank shape, indices and all eight pinned permutations.

Galvani A100 smoke 3129779 passed at 19:40:19 CET. Array 3129788 completed all 59 additional fits. CPU collector 3129789 finished at 19:47:42 CET. Release `/mnt/lustre/work/gehler/gfh527/ni26_friday_ea_shallow_fresh_20261118`, code `ac43349`.

| Artifact | SHA-256 |
| --- | --- |
| `source.sha256` | `c1b4d6f86640cebd61cffbf43740950fa37ec7c86b4c598d254c7be83b27bb9f` |
| `fresh_bank_receipt.json` | `c48d4f46f8283282a78f88e27c7a86a5b4c98b51f512d0d10119143e61a1be60` |
| `shallow_ea_fresh_all_loso_orders.npz` | `9cd2921638e7e6496a7b849f1bdfe131792a35d17bbc836c2bcfee0a448a1048` |

The bank contains canonical `index`, six `seeds`, eight `orders`, their global `delivery` rows, `proba (6, 8, 2655, 3)` and `recording (6, 2655, 3)`. These weights are new. The historical shallow reconstruction remains unresolved, and the existing raw Shallow slot remains a separate artifact. Replace the shared input to both comparators and replay all dependent causal state and stages before interpreting an accuracy difference. This release contains no downstream accuracy result.
