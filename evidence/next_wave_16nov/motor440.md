# §440 Waveform motor evidence in the full cap250 decoder

The phase-trained motor insertion reduced full-system new-person development accuracy from **76.279% to 75.905%**. Its paired loss was **−0.374 pp** (80% person/order interval −0.546 to −0.132). Equal-person weighting gave −0.218 pp (−0.381 to −0.053). Both upper bounds fell below zero, meeting the preregistered refutation condition for this fixed 10% insertion.

| Decoder | Accuracy | Motor recall | Subtraction recall | Word recall |
|---|---:|---:|---:|---:|
| Stock cap250 | 76.279% | 81.529% | 75.305% | 72.004% |
| Real-only motor insertion | 75.885% | 80.395% | 75.481% | 71.778% |
| Phase-trained motor insertion | 75.905% | 80.886% | 75.164% | 71.664% |
| Real-only all-class insertion | 75.515% | 80.290% | 75.188% | 71.066% |
| Phase-trained all-class insertion | 75.774% | 80.961% | 74.945% | 71.416% |

Compared with the matched real-only motor insertion, phase training raised full-system motor recall by 0.491 pp and overall accuracy by 0.020 pp (−0.186 to +0.197). The equal-person difference was −0.059 pp (−0.245 to +0.122), with 5/10 people improving. P04/P10 differences were +0.107/+0.330 pp against real-only motor and −0.690/−0.843 pp against stock. With all three posterior classes blended, phase training improved over real-only by 0.260 pp (+0.066 to +0.419), equal-person +0.199 pp (−0.002 to +0.378). The phase-trained all-class decoder still lost 0.505 pp against stock.

## Replay and scope

Each model trained on nine source people with the original §435 sequence, initialization and intervention. All ten refits reproduced **both** original decision vectors and the training episode hashes. A temperature-one softmax converted the recovered logits to probabilities. The motor insertion mixed only the normalized stock and member motor probabilities, preserving the stock subtraction-to-word conditional distribution to 1e−12 at the insertion. Both arms then passed through the original count filter and CP. These downstream stages can change the cognitive decisions.

The score covers 2,655 windows, 14 sessions, 28 official cells, ten held people and eight independently verified shuffled orders. The zero arm reproduced every stock probability after count filtering and CP and every archived native stock decision. The separate recording-order replay printed OK. All intervals resample people and orders. The public cohort previously supplied the class-recall observation that motivated this test. No sealed evaluation, returning-person evaluation or package change.

Galvani source smoke 3125339, ten A100 folds 3125340, actor smoke 3125503, eight order workers 3125504 and collector 3125505 all completed. Code revision `28670a3`; §435 initialization seed 379 and phase generator seed 435. [Machine-readable metrics](motor440_result.json) and [pre-score actor seals](motor440_before.json).
