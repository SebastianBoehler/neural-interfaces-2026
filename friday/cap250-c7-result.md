# F-C7 results — Learn arbitration on PROTEUS and test exact Scherer

Retain exact staged v6.6cap250. The learned selector scored **48.4028% FULL-SYSTEM EXTERNAL**, compared with **48.4557%** for exact cap250. The rule fixed eight stock errors and introduced ten errors across 20 changed decisions. Its primary comparison remained undecided, and it underperformed the F-C6 causal margin rule on this instrument.

| Full-system external comparison | Change, pp (paired 80% interval) | Changed decisions |
|---|---:|---:|
| Learned − exact cap250 | −0.053 [−0.228,+0.132] | 20 |
| Learned − source-permuted control | −0.053 [−0.228,+0.132] | 20 |
| Learned − F-C6 causal rule | −0.261 [−0.420,−0.116] | 22 |
| F-C6 causal rule − exact cap250 | +0.208 [+0.023,+0.394] | 24 |
| Internal incumbent − exact cap250 | −0.493 [−1.283,+0.314] | 384 |

The source-permuted rule selected the stock label on every external delivery. The F-C6 causal rule is a prespecified comparator. It fixed 14 errors and introduced five. Its two order differences were 0.000 and +0.417 pp. The nine people were excluded from fitting both branch decoders and the selector. Their external scores come from an instrument examined in earlier studies. The direct PROTEUS reverse-direction evaluation requires fresh decoder fits that exclude the public evaluation people, because the production feature generators have already used their labels.

The exact staged solver exposed both branch scores on the normal one-window path. Its terminal decisions reproduced every archived cap250 label and all activity counts on 4,260 deliveries. The inner incumbent reproduced all 4,260 standalone v5.1b labels. Batch and streaming causal features agreed, and a separate fresh 17-window prefix reproduced every smoke prediction. No held labels configured the rules. All nine decision files were authenticated before opening external labels.

Source smoke 3128118, fit 3128121, exact A100 smoke 3128122, nine-worker array 3128141, and scorer 3128142 all finished 0:0. The full A100 tasks took 2m30s–2m45s each, and collection took 12 s. Historical replay printed OK. Source manifest `7927abf1d34bbe8b35b5d5a03dd30dd64e4f33461cd44341ed5bd0714baf4e22`. Coefficient artifact `fb71ab5bac6b3b1235b6464338bed699c86cf247d095fd2f3b11d153890fea81`. Before-label receipt `708eec13861706bcdfb50430100bec44504436a129606275cdcf3a3c9173b3af`. [Paired results per person and order](cap250-c7-result.json) · [Registration and fit provenance](cap250-c7.md) · [Smoke and queue](cap250-c7-queue.md)
