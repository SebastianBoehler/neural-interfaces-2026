# §406 Finite-prefix count filtering

The staged cap250 Python filter checks its lag statistic at delivered indices 40, 50, 60, 80, 100 and 150. After a successful check it keeps filtering the class probabilities, but later EEG views cannot influence the latch. After all six checks miss, the stock filter returns each input probability unchanged. The private variant stops constructing EEG views at either terminal decision and skips the urn update when every check missed.

## Matched conditional returning replay

Source 09356be, deterministic CPU seed 3211701, §321 fold-specific original-class weights and retained production priors. The fold package and the submission.py entry in the staged ZIP share SHA-256 37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f. Gold labels were unnecessary. P04/P10 contributed four sessions and 762 unique windows, each delivered in eight pinned orders. Every filter output, latch and archived final package probability/decision matched bitwise. All six recording-order v3 references passed separately.

| Session-orders | Count | Stock views | Retained views | Median reduction in CPU filter wall |
|---|---:|---:|---:|---:|
| All | 32 | 6,096 | 2,120 | 57.76% |
| Latch engaged | 25 | 4,760 | 1,070 | 58.03% |
| All checks missed | 7 | 1,336 | 1,050 | 20.58% |

Stock and pruned filters first processed the actual stream together in the complete package. Each then replayed the same EEG and upstream probabilities in warm AB/BA order. Across both timed repeats filter wall totaled 114.41 s stock and 57.80 s pruned. The preregistered 25% median reduction held across all sessions. This measures finite real public inputs in the CPU worker image. Whole-worker runtime, sealed data, changed accuracy and behavior under malformed inputs remain unmeasured.

Pilot 3124211 stopped before inference because the remote checkout lacked the local submission source path. The corrected pilot checked the actual package against the staged source digest. Galvani pilot 3124218, array 3124219 and collector 3124234 completed. [Per-session warm pairs, order and input archive hashes](filter406_result.json).
