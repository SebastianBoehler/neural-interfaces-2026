# F16e raw-only S1C on Scherer

Both registered transfer effects remain undecided. With bounded inference, trial distillation scored 45.9921% and 44.2659% across seeds 0 and 1. The matched hard CE control scored 45.9392% and 44.5437%. Scherer contributed no fit rows. The assessment reused the nine §411b/F15e Scherer participants, two visits per person and delivery orders 411 and 1411. [Preregistration](../../docs/LEDGER.md#f16e--external-validation-of-zero-feature-distillation)

| WHOLE EXTERNAL DEV balanced accuracy | Seed 0 ordinary | Seed 0 bounded | Seed 1 ordinary | Seed 1 bounded |
|---|---:|---:|---:|---:|
| Hard CE, zero fixed features | 45.2216% | 45.9392% | 44.3353% | 44.5437% |
| Trial-specific KD, zero fixed features | 44.9306% | 45.9921% | 44.2229% | 44.2659% |
| Within-class/session deranged KD, zero fixed features | 44.6627% | 44.9173% | 43.8856% | 43.9815% |
| Contextual F15e dropout teacher, full fixed features | 41.8221% | 42.1925% | 40.7341% | 40.5754% |

| Paired contrast in percentage points [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded trial minus hard | +0.0529 [−0.9954, +1.0880] | −0.2778 [−1.0648, +0.5093] |
| Bounded trial minus deranged | +1.0747 [−0.2646, +2.3611] | +0.2844 [−0.2880, +0.8565] |
| Bounded deranged minus hard | −1.0218 [−2.0139, −0.0331] | −0.5622 [−1.2500, +0.1190] |
| Ordinary trial minus hard | −0.2910 [−1.0284, +0.4663] | −0.1124 [−0.8763, +0.6184] |
| Ordinary trial minus deranged | +0.2679 [−0.9755, +1.4583] | +0.3373 [−0.3373, +1.0086] |
| Ordinary deranged minus hard | −0.5589 [−1.4253, +0.3307] | −0.4497 [−1.0483, +0.1455] |

The trial arm stayed above the deranged arm on both orders at both seeds. All four intervals against derangement crossed zero. The raw hard CE point scores exceeded both matching full-feature teacher point scores. This contextual comparison changes training and inference together. [F15e teacher evidence](../fbc323_skip_external/result-18nov.md)

The teachers predicted their own PROTEUS source rows, with per-session restart and prefix checks. The two source schedules preserved all 217,600 padded teacher target appearances under the donor control. Active KL gradients reached the waveform body and classifier in both KD arms. Scored inference used the waveform representation and a constant zero vector. All 2,130 external labels matched the pinned native codes and cues after decision sealing. The nine held banks passed 216 exact first-window and 432 exact restart checks. Maximum prefix posterior difference was 1.367e−6. The paired intervals resample people, visits and orders.

Ferranti source smoke 511160 passed in 12 seconds. Both 511161 production allocations passed in 37 seconds each. Galvani smoke 3131549 passed in 20 seconds. All nine 3131550 replay tasks passed, and collector 3131551 completed 18 November 2026 at 21:41:17 CET.

| Galvani release `ni26_friday_s1c_distill_external_20261118` | SHA-256 |
|---|---|
| `actor.sha256` | `b4dd2f3eb7ae50c417334fed30d4a5f06553e28631892245e0e456983601441a` |
| `score.sha256` | `306c0a5fe64e1c8254262c97fe4a7c8032b028e3acc1fb87a5a5cc798dc57d98` |
| `before_labels.json` | `bc383cbc561223a803ef597300de5cde5869034dc35e464f0852462a2593ca94` |
| `result.json` | `75cbbaf0d2df958cb2a353c036629e6dc0bc9253465d5e30156833d428234a5f` |
