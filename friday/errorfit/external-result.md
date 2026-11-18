# F5 external evaluation of waveform corrections trained on cap250 mistakes

**Keep staged cap250 ahead of F5.** Both fitted waveform corrections reduced external balanced accuracy. Preserving the incumbent's actual incorrect-class scores helped relative to swapping them. That improvement did not recover stock performance.

| Full-system external comparison | Seed 460, pp [paired 80%] | Seed 1460, pp [paired 80%] |
| --- | ---: | ---: |
| True teacher − staged stock | −3.284 [−4.160, −2.338] | −3.975 [−5.033, −2.873] |
| True teacher − swapped wrong-class scores | +1.366 [+0.694, +2.037] | +1.009 [+0.493, +1.515] |
| True teacher − ordinary waveform CE | +0.093 [−0.169, +0.351] | −0.747 [−1.346, −0.139] |
| True teacher, genuine − blank history | −0.724 [−1.111, −0.324] | −1.951 [−2.814, −1.101] |

Stock scored **48.456%**. The true teacher scored **45.172% and 44.481%** with genuine history, and **45.896% and 46.432%** with blank history. The genuine arms changed 1,026 and 1,211 of 4,260 decisions. They corrected 298 and 350 stock errors, and regressed 438 and 519 stock hits. The preregistered two-seed rule refutes a stock gain and supports the contrast against swapped wrong-class scores. The contrast against ordinary waveform training remains undecided under that rule. [F5 preregistration](../../docs/LEDGER.md#f5--train-a-waveform-correction-on-fit-excluded-cap250-mistakes)

Training used all ten public PROTEUS participants and their historical fit-excluded cap250 predictions. The external test used nine separate Scherer participants, two days each and both canonical orders. This Scherer cohort has supported earlier project experiments. Intervals resample participants, days and orders. The test instruments the three external classes on unrecognized routes. Private performance and returning-person performance remain unknown.

Galvani 3127363 verified the worker image, pinned evaluator and frozen archive. Historical replay printed `OK`. A100 smoke 3127364 reproduced all 120 archived labels and adaptation counts for Scherer G day 1, order 0. Its independent fresh-state replay matched all arms over the first 17 windows. All nine A100 workers in 3127365 completed. The collector 3127366 authenticated the blind bank before opening labels and reproduced the official metric on 18 cells per order. The full test matched stock labels and work counts on all 36 streams. A100 allocation time was 31m08s across smoke and production.

The median warmed one-window increment across stream medians was **1.367–1.373 ms per waveform actor** in the same worker process, including both genuine and blank readouts and their CPU result transfer. The paired stock-call median was **162.64 ms**. The timing excludes model loading and measures branch cost inside a single shared stock pass. The first actor incurred cold startup overhead, which raised its total to 17.55 s over all windows. Other actors took 6.67–6.71 s.

The owning Galvani release is `ni26_friday_errorfit_20261118`, evaluator commit `7fc5a9f`. All six Ferranti production fits in 510111 completed after smoke 510110. The teacher bank SHA-256 is `df80f1d5cbb514e750beeeb84f953fb16d3fa4eec5aa98ad21949b11602f4435`.

| Galvani artifact | SHA-256 |
| --- | --- |
| `external_source.sha256` | `a58224dd513a8149d88154f55e9a166403cb95d88d865ba332e499b7b6a8def9` |
| `weights/models.sha256` | `cc7a9d0c51de6ccfaf8649da98dc8cb64c613c6febd6c7a37015c331419fd89a` |
| `external_predictions.npz` | `232d6fe335c3a350a67c55e4adfec78b2da47eb9f7a76d716bc23c3ee4a5310b` |
| `external_before_labels.json` | `d51cb0a1558a49ff298a119be7ec987eca91f480ad5742ddff596d3d834470ac` |
| `external_result.json` | `f74613ea90c7ae758f497eb8dfce5e62bce8b1f6a61e0c9a3503d3571eabc511` |
