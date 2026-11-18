# F14 bounded causal training in S1C

Bounded causal training and bounded evaluation scored **56.5091% and 56.2163% held-new WHOLE DEV**. The same-platform original S1C controls scored **58.9325% and 59.4776%**. Both paired intervals fall below zero, refuting the registered positive whole-decoder prediction. With bounded evaluation fixed, bounded causal training remained indistinguishable from ordinary causal training under the registered rule.

| Source training | Ordinary held evaluation, seeds 0 / 1 | Bounded held evaluation, seeds 0 / 1 |
| --- | ---: | ---: |
| Original S1C | 58.9325% / 59.4776% | **59.2454% / 60.2583%** |
| Causal training | 56.1740% / 56.1442% | 56.4354% / 56.4455% |
| Bounded causal training | 56.0290% / 55.8065% | 56.5091% / 56.2163% |

| Paired contrast | Seed 0, pp [80%] | Seed 1, pp [80%] |
| --- | ---: | ---: |
| Bounded-training effect at fixed bounded evaluation | +0.0737 [−0.3533, +0.4976] | −0.2292 [−0.5475, +0.1561] |
| Bounded trained and evaluated versus original S1C | −2.4234 [−3.7571, −0.8227] | −3.2613 [−4.3850, −1.7080] |
| Causal versus original training at ordinary evaluation | −2.7585 [−4.3549, −0.8300] | −3.3334 [−4.0142, −2.3553] |
| Bounded evaluation on original S1C weights | +0.3129 [−0.4484, +1.4194] | +0.7807 [+0.0382, +1.8292] |

The full bounded change lost 2.1384 and 4.5480 pp on P04, and 6.5184 and 6.1433 pp on P10. Under identical bounded evaluation, the training-only differences were −0.9437/−0.2777 pp for P04 and +1.0205/−1.1499 pp for P10. The official comparison weights participant/session/interface cells equally within each of the three delivery schemes and resamples people and deliveries within those schemes. The historical S1C aggregate of 59.0453% used the earlier Galvani training release. Today's primary control used the same Ferranti H100 environment as both new training arms. [Historical architecture and matched F13 replay](../fbc323/result-18nov.md)

All three variants used the original architecture, 347,903 trainable parameters, identical 40-epoch source schedules and 1,320–1,600 optimizer steps per fit. Every bounded fit clipped 499,365–717,786 past-channel terms. The independent CPU replay authenticated all 60 checkpoints, all 672 first-window identities and the session reset checks. The maximum prefix posterior deviation was 1.286e−6. Twenty decision bank hashes were sealed before opening labels.

Ferranti 511007 caught the smoke probe changing the initial model fingerprint. The accepted probe used a separate copy. One-epoch smoke 511008 passed the numerical source checks and stopped at its activity gate. Four-epoch smoke 511010 passed in 11 s. Production array 511011 completed all 20 triples at 30–41 s per successful allocation. The P02/seed0 node failure requeued automatically and the successful retry finished at 20:09:38 CET. Galvani held smoke 3130019 and full replay 3130020 passed. Collector 3130021 finished on 18 November at 20:11:46 CET.

| Artifact | SHA-256 |
| --- | --- |
| Ferranti actor manifest | `74b2c7deeffd04d97a1355a437bc1e8704970d5f8e2ffbf426594ff0cf0b9824` |
| Ferranti runtime lock | `da21360c9a85d7561eee8f0f68367b545797641a69e3726354275cbf404ec30e` |
| Galvani scorer manifest | `f9171d7228cce33215807cad7988438dd3e3879d4d3b1620e3cec4c17ea32386` |
| Blind held decision manifest | `b012b681c8bac77a11a83ce07dd9e2b171ce4eb33ca322a175de24afdcaa7920` |
| Official score receipt | `82fe550621a8eb72a0762c507fb81ee1a9d4d3cc65983a3e8bbdf2d923aed420` |

Ferranti release `~/ni26/friday_s1c_training_20261118`. Galvani release `ni26_friday_s1c_training_20261118`. [Preregistration](../../docs/LEDGER.md#f14--train-the-original-s1c-architecture-with-bounded-causal-normalization)
