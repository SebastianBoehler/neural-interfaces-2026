# Arm c returning-side sprint · 21 November 2026

**22:20 CET · Upload arm c.** All three candidates missed at least one promotion criterion. The on-disk ZIP matches SHA-256 `cba9b208fbff7ce416efd8b5ce68ae1ad0771980d84e754433bd33e9996d0bca`. Each candidate changed **0/63,720** held-new decisions. [submission.zip](../../submission.zip) · [Arm c qualification](arm_c_confirmation.md)

EM0 set the recognised SessionEM output likelihood weight to zero and retained its history and fit. RB50 halved RouteB Rebalance alpha after recognition. Route150 multiplied the native winner's ramp by 1.5 up to 1.0 after recognition. RB50 and Route150 kept arm c's 0.25 SessionEM weight. All candidates ran with native TSEM, CountFilter, CSP and MANAS after the intervention. [Preregistration](../../docs/LEDGER.md#gpt-sprint--arm-c-returning-side-extensions-21-november-preregistered-at-2147-cet)

## PROTEUS

Balanced-accuracy deltas in percentage points versus arm c. The 50/50 estimates give equal weight to the held-new and returning cohorts. Intervals use 20,000 paired draws that resample people and then orders. Both order sets reuse the two returning people and ten held-new people.

| Orders | Delivery | EM0 50/50 [80%] | RB50 50/50 [80%] | Route150 50/50 [80%] |
| --- | --- | ---: | ---: | ---: |
| 71808–71815 | Full-session | -0.043 [-0.267, +0.184] | -0.129 [-0.274, +0.009] | +0.123 [-0.041, +0.289] |
| 71808–71815 | Within-run | -0.080 [-0.214, +0.057] | -0.063 [-0.144, +0.019] | +0.024 [-0.041, +0.089] |
| 72120–72123 | Full-session | +0.037 [-0.113, +0.204] | +0.259 [-0.096, +0.569] | +0.082 [+0.000, +0.214] |
| 72120–72123 | Within-run | -0.066 [-0.183, +0.046] | -0.163 [-0.363, +0.016] | -0.017 [-0.068, +0.033] |

Route150 passed the fresh full-session check and missed the held lower-bound requirement. Its within-run points were +0.024 and −0.017 pp on the held and fresh orders. The live smoke verified that the ramp changed RouteB posteriors when it had room to increase. The held and fresh cohorts showed no mistaken recognition.

| Orders | Delivery | Arm | Returning Δ [80%] | P04 / P10 Δ | Returning labels changed | Active streams |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 71808–71815 | Full-session | EM0 | -0.086 [-0.533, +0.367] | +0.335 / -0.508 | 90/6,096 | 32/32 |
| 71808–71815 | Full-session | RB50 | -0.258 [-0.547, +0.018] | -0.090 / -0.425 | 164/6,096 | 32/32 |
| 71808–71815 | Full-session | Route150 | +0.246 [-0.082, +0.577] | +0.559 / -0.067 | 58/6,096 | 22/32 |
| 71808–71815 | Within-run | EM0 | -0.159 [-0.428, +0.113] | +0.064 / -0.383 | 62/6,096 | 32/32 |
| 71808–71815 | Within-run | RB50 | -0.125 [-0.288, +0.039] | -0.049 / -0.202 | 119/6,096 | 32/32 |
| 71808–71815 | Within-run | Route150 | +0.049 [-0.081, +0.178] | +0.163 / -0.065 | 20/6,096 | 19/32 |
| 72120–72123 | Full-session | EM0 | +0.073 [-0.226, +0.408] | +0.338 / -0.191 | 37/3,048 | 16/16 |
| 72120–72123 | Full-session | RB50 | +0.517 [-0.191, +1.139] | +1.135 / -0.101 | 91/3,048 | 16/16 |
| 72120–72123 | Full-session | Route150 | +0.164 [+0.000, +0.428] | +0.328 / +0.000 | 21/3,048 | 9/16 |
| 72120–72123 | Within-run | EM0 | -0.131 [-0.365, +0.093] | -0.200 / -0.063 | 34/3,048 | 16/16 |
| 72120–72123 | Within-run | RB50 | -0.327 [-0.726, +0.031] | -0.595 / -0.059 | 60/3,048 | 16/16 |
| 72120–72123 | Within-run | Route150 | -0.034 [-0.136, +0.065] | -0.132 / +0.065 | 9/3,048 | 9/16 |

## Scherer B-ret

The Scherer-free variant replayed nine returning people with visit 1 enrolled and visit 2 streamed on orders 71864–71871. Galvani used the native A100 BF16 path. Arm c matched archived A100 control labels in every shard. Each scheme used 72 returning streams and 8,400 decisions.

| Arm | Full-session returning Δ [80%] | Within-run returning Δ [80%] | Changed labels full / within | Transfer |
| --- | ---: | ---: | ---: | --- |
| EM0 | +0.261 [+0.071, +0.453] | +0.091 [-0.202, +0.387] | 127 / 152 | Pass |
| RB50 | -0.298 [-0.769, +0.159] | -0.721 [-1.334, -0.107] | 502 / 549 | Miss |
| Route150 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0 / 0 | Pass |

## Checks and receipts

- Leonardo smoke `65063031_11` printed v3 `OK`. All three patches matched live 84-window labels and RouteB/terminal posteriors within 2e−5 on P10 visit 3 in both schemes. The prefix replay was exact. Held returning `65063032`, fresh returning `65063033` and held-new `65063034` completed all 104 scheduled tasks.
- The initial Leonardo collector iterated over the control as if it were a candidate. The corrected collector used all Leonardo shards locally, verified fold receipts, independently regenerated the delivery rows, and reproduced the archived full-session and fresh arm c controls.
- Galvani smoke `3150681`, nine B-ret tasks `3150682` and collector `3150700` completed. The smoke matched live labels for all four policies and the collector independently regenerated all delivery rows.
- Galvani backup `3150746`/`3150747` checked 40 held-new fold/order shards. It changed no labels against its own arm c over 10,620 windows. Arm c controls on the two clusters differed at 172/10,620 decisions on identical rows. The PROTEUS report uses Leonardo for every shard.
- PROTEUS receipt SHA-256 `f1d8e2055f33b3862cc12b61c8e887ade80ebd28182dd9f97072cc8f35b968a0`. Scherer receipt SHA-256 `87582dd2bb4500f9305c7e07d6fb3b8d20e2c9622e40fce096c9ff34973c4efd`. [PROTEUS JSON](../../work/arm_c_sprint/local-out/proteus.json) · [Scherer JSON](../../work/arm_c_sprint/scherer.json)
