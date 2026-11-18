# Teaching waveform memory about session-wide electrode gains

On ten held public people, coherent augmentation raised independent waveform-decoder accuracy from 41.692% to 43.923%. Nine improved. The official paired gain was +2.231 percentage points [80% +0.982,+3.432]. Using the same electrode gains independently across windows scored 42.907%, giving coherent minus independent +1.017 [+0.144,+2.031]. Seven people improved on this second comparison. Every evaluation window contained original EEG.

Each fold trained on nine people. Class-balanced source-session log RMS supplied the gain patterns. Half of 360 shuffled episodes received a single pattern across all 96 windows; the window-independent control permuted exactly those gains within 12-episode blocks. All three decoder types used the same training episodes and initialization as the pinned genuine-only experiment. The people had already contributed to earlier research evaluations.

| Training | Static decoder | Explicit normalization | Learned memory | Same weights, blank | Same weights, donor |
|---|---:|---:|---:|---:|---:|
| Genuine EEG | 44.120% | 44.149% | 41.692% | 41.237% | 43.034% |
| Independent electrode gains | 44.222% | 44.828% | 42.907% | 42.401% | 44.587% |
| Session-wide electrode gains | 44.888% | 44.716% | 43.923% | 42.160% | 44.966% |

With coherent training, genuine history beat blank by 1.764 pp. Relative to genuine-only training, that difference increased by 1.308 pp [+0.391,+2.283]. The improvement relative to the matched-age donor was only +0.299 pp [−0.670,+1.250]. Blank simultaneously removes the memory write and the age input. The matched-age donor retains both and comes from one selected source session per fold. This result identifies an augmentation effect. It leaves open whether the decoder needs the target session’s individual EEG history.

The pinned full decoder scored 76.279% on these folds. §441 repeats the training from a new initialization and separates write content from age. §442 tests transfer to the stronger A6 model. §443 tests a contribution to the complete decoder. Replay printed OK. Galvani allocations 3125269, 3125274 and 3125275 consumed 726 allocated CPU-core-seconds and 279 A100-seconds across 22 allocations. [All scores and per-person contrasts](gain439_result.json) · [Source](../../studies/18nov/gain439/fit.py)
