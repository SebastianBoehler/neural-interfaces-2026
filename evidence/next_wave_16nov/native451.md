# Production C++/CUDA and staged Python on Scherer EEG

The C++/CUDA Solver reproduced **4,260 of 4,260** decisions from the staged Python cap250 Solver. Each also matched the archived §385 Python predictions. All ten work counters matched on every stream. The two implementations therefore have the same historical external three-class balanced accuracy, **48.456%**, on these nine participants. This run read no task labels. The 4,260 deliveries represent 2,130 distinct windows in two orders. [Full comparison](native451_result.json) · [Archived score](qualification385_result.json)

| Complete canonical workload | C++/CUDA | Staged Python |
|---|---:|---:|
| Naturally unrecognized session/order streams | 36 | 36 |
| Neural refits / relabels / Adam steps | 356 / 356 / 7,120 | 356 / 356 / 7,120 |
| Personalization refits / fusions | 180 / 1,020 | 180 / 1,020 |
| Cap skips / time skips / stage errors | 0 / 0 / 0 | 0 / 0 / 0 |
| Summed Solver stream time | 1,292.311 s | 1,445.968 s |

The median within-participant C++/Python time ratio was **0.92362**, a 7.64% saving. Eight of nine pairs were faster. Individual ratios ranged from 0.6993 to 1.0011. The 153.657 seconds saved across canonical streams correspond to 10.63% of Python time. Each participant's two implementations ran in separate clean processes on one A100, with alternating arm order across participants. Stream time includes fresh-model copy, delivery, prediction, disposal and CUDA synchronization. It excludes package load and container startup. Each pair ran once. Concurrent node workloads and hardware differences across allocations limit the timing precision and private-worker extrapolation.

**Process history.** After four canonical streams, each implementation repeated day 1/order 0 from a fresh official model copy in its warmed process. F retained every label. G changed index 104, original row 1216, from WORD to SUB in **both** implementations. Their warmed outputs still matched on all 240 deliveries. All repeat counters matched. The fresh model copy did not remove this process-history dependence. These comparisons do not identify its numerical cause.

**Provenance.** The measured §366 ordinary package matched all 43 runtime files in the qualified release, manifest SHA-256 `d33cfb35de9944bcb3d3b703091a12c6fe2a6b4eb6f900a6396d1d28079fe925`. All 37 shared PT assets matched the staged Python package byte for byte. The staged ZIP SHA-256 was `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`. Both arms saw the original exported sensor windows and pinned §385 permutations, with the production recognition thresholds and unchanged weights. Galvani smoke 3126282, nine producers 3126283, collector 3126291. Project replay printed OK. Source revision `76d6d95`. Numeric-result SHA-256 `66fc2fee3638bb82865d70098435e8f9c5562750dfe4e4f4bc8501cf880fdaff`. The previously used Scherer instrument includes 19 measured and 22 reconstructed target channels and different motor imagery. No sealed score or time is available for this C++ package. [Worker](../../studies/18nov/native451/worker.py) · [Comparison](../../studies/18nov/native451/collect.py)
