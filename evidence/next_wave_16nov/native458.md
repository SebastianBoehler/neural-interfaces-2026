# Native two-seed causal kernel on one delivered EEG window

The C++/ATen implementation classified two frozen spectral memory models together on one A100. Its fixed-shape CUDA graph cut the median increment for both models from 1.24–1.26 ms to 0.17 ms per window. This measures an optional decoder from pinned CPU EEG input through the return of both corrections to CPU. Package loading, graph capture, stock Solver work and other worker costs are outside the timed region. There is no batching of future EEG. Only model parameters and two independent session-local histories are batched.

| Real shuffled stream (within0) | Windows | Serial C++ | Batched C++ | Batched graph | Graph/serial, paired |
|---|---:|---:|---:|---:|---:|
| P04 session 1 | 189 | 1.256 ms | 0.787 ms | 0.170 ms | 0.136 |
| P10 session 1 | 192 | 1.244 ms | 0.782 ms | 0.169 ms | 0.136 |

Each row contains three rotated warm pairs. Both methods use one shared FFT feature pass, source-only standardization and the same two immutable H100-trained checkpoints. The code reads each state before adding its current write. It resets at every repetition. All raw genuine/source logits reproduced §454 within 1.20×10⁻⁶. First-window corrections were exactly zero for each model and nonzero later. The development driver tested the Galvani LibTorch environment; it did not load a competition ZIP. Ordinary pageable application inputs or cold capture can add latency.

## FULL-SYSTEM DEV, historical three-eye/three-rest cap250

Ten held source-excluded people, 28 official cells, eight pinned shuffled orders. The two-seed mean scored **76.2808%** against stock **76.1039%**. Public-development confidence intervals resample people and orders and reflect the reused cohort.

| Two-seed mean minus comparator | Difference in percentage points | Paired 80% interval |
|---|---:|---:|
| Stock cap250 | +0.1769 | [+0.0800, +0.2652] |
| Two-seed matched donor | +0.1859 | [+0.0210, +0.3405] |
| Single seed 454 | −0.0406 | [−0.0928, +0.0084] |
| Single seed 1454 | +0.0186 | [−0.0217, +0.0609] |

The two-seed mean improved stock for nine people. Its point estimate falls below the stronger single seed. The predicted benefit over that seed remains unsupported. The scorer reproduced both §456 single-seed genuine, donor and stock decisions for every delivery and committed blind predictions before reading labels. It used archived H100 scores to test the complete historical system. The separately timed C++ graph was validated on two streams. Neither test measures sealed accuracy.

Source commit 0b94319. Galvani build 3127014, smoke 3127015, paired timing 3127016, real scoring smoke and project replay 3127017, final score 3127018. [Timing pairs, provenance and individual effects](native458_result.json)
