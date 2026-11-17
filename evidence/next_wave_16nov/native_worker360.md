# Current-source native worker benchmark

The reference is frozen v6.6cap250. The native arm rebuilds commit
`a9e307d` with the TorchScript deepcopy and shared Adam repairs. CPU job
3122062 built and installed both portable libraries in 473 seconds. The
native package manifest is
`f5ae55510adc986337bb1f56e3e25072413fd8b51194b2f8a2ce9f5716c50cb1`.

## A100 preflight

Job 3122082 ran both adapters inside the worker replica in separate
processes. Two fresh copies per arm exactly matched all labels and
adaptation counters. Both stress arms performed five fine-tuning refits
with 100 optimizer steps and three personalization refits per repeat.
The matched 31-window stream recognized at index 10.

| Workload | Part | Frozen | C++ |
| --- | --- | ---: | ---: |
| 31 normal windows | Model load | 4.73 s | 4.45 s |
| 31 normal windows | First replay | 24.25 s | 23.54 s |
| 31 normal windows | Warm fresh repeat | 4.58 s | 5.19 s |
| 64 forced-unrecognized windows | Model load | 4.70 s | 4.51 s |
| 64 forced-unrecognized windows | First replay | 47.22 s | 43.82 s |
| 64 forced-unrecognized windows | Warm fresh repeat | 13.89 s | 15.51 s |

Normal load plus both replays took 33.56 s frozen and 33.18 s native.
Stress load plus both replays took 65.81 s frozen and 63.85 s native.
The warm repeats were 13.3% and 11.7% slower on native. The differing
cold costs and short stream lengths make these checks unsuitable for
budgeting time in a complete evaluation.

## Complete workloads

Job 3122084 replays all 2,655 public EEG windows in 14 shuffled sessions.
All labels and work counters matched. All 14 sessions recognized at index
10, so the fine-tuning and personalization refits stayed inactive. The
job finished in 931 s on an A100 with two allocated CPUs. The first replay
includes cold compilation. The remaining 13 sessions showed the same
direction of timing difference.

| Measurement on the same A100 | Frozen | C++ |
| --- | ---: | ---: |
| 14-session solver replay | 415.083 s | 449.866 s |
| Model load | 4.641 s | 4.761 s |
| Process wall from public input load | 420.168 s | 455.067 s |
| Replay throughput | 6.396 windows/s | 5.902 windows/s |
| Peak resident host memory | 2,329,612 KiB | 2,280,648 KiB |
| Peak live CUDA allocations | 43,390,464 B | 38,312,960 B |

C++ uses 34.783 s more solver replay time, an 8.38% increase and a
7.73% reduction in throughput. At the same mix of work, an hour of frozen
solver replay would take 65.0 minutes in C++. This conversion excludes
sealed worker overhead and says how much work fits in an hour at the
measured rates. The challenge supplies a fixed set of input windows.

Job 3122085 replays 378 and 800 delivered windows with recognition thresholds
raised on both arms. This engineering workload covers active personalization,
fine-tuning and the cap. A single-window prediction gets only the current
tensor in both cases. Gold stays outside the timing run.

The long adaptation pair failed exact label parity. Both solvers performed
24 fine-tuning refits, 480 optimizer steps and 24 relabels in each stream.
The shorter and longer streams had 18 and 39 personalization refits, 288
and 710 personalization fusions and 12 and 54 skipped refits at the cap.
The stage logs reported zero errors.

| Forced-unrecognized windows | Frozen | C++ | Changed labels |
| --- | ---: | ---: | ---: |
| 378 | 151.377 s | 150.772 s | 1 at index 297 |
| 800 | 153.922 s | 162.811 s | 4 at indices 312, 405, 602, 612 |
| Total replay | 305.300 s | 313.583 s | 5 across the two runs |

Both solvers see the same first 378 inputs in each run. Frozen changes its
label at index 297 between fresh sessions. C++ changes its label at
index 312. Exact parity failed despite the same work counts.
Job 3122389 repeated the first 330 inputs twice per arm on a second A100
with explicit deterministic CUDA settings. Each arm and repeat reproduced
its corresponding default-worker prefix exactly. The cold cross-arm
disagreement remained at index 297, and the warm disagreement at 312.
Job 3122401 disabled TorchScript graph optimization in both arms.
All 330 labels matched across the two solvers and both fresh repeats.
Each repeat still performed 24 fine-tuning refits and 16 personalization
refits. This identifies graph execution sensitivity in the measured
prefix. Default-worker long-stream parity remains open.

Each pair runs on one A100 and includes copies and resets, B1 transfers,
CUDA synchronization, and stream teardown. The report records label and
adaptation-count parity, measured savings per workload, load time and
peak memory. The lab uses prepared public windows. It measures solver
throughput and excludes the sealed worker's data preparation.

For matched replays of R seconds and N seconds, the observed saving is
R − N seconds. At a fixed evaluation budget and the same repeated workload
mix, model inference capacity scales by R/N after setup. A sealed-time
estimate also needs the sealed workload mix and its measured overhead.

## Native iterations

Job 3122090 stacks all neural posterior rows on the GPU and transfers them
once per input, then validates finiteness on the host. On the same A100,
578 outputs and all activity counters match in each pair.

| Pair order | Original native | Batched transfer | Replay saving |
| --- | ---: | ---: | ---: |
| Original → batched | 112.547 s | 111.479 s | 0.948% |
| Batched → original | 112.200 s | 111.691 s | 0.454% |

The Slurm job finished all four replays in 542 s, then failed when its
collector looked for a stress file. The current normal-only comparator
revalidated the stored outputs after checking the manifests and stage logs.
The change missed the 3% paired threshold. Production keeps the original bank.

Job 3122137 runs the two independent tangent cores concurrently. It
changes only scheduling. All labels and counters match across the same
three sessions and both orderings. Original → parallel saves **5.657%**
(116.607 → 110.011 s); parallel → original saves **1.386%** (112.366 →
110.809 s). The second pair misses the 3% threshold. Production keeps
the existing scheduling.

The channel-parallel filter groups four independent rows so the
compiler can vectorize the recurrence. Job 3122398 stopped when the
original draft failed the exact numerical probe. That draft had read
the start sample after each filter section instead of once per pass.
The corrected job 3122414 checks the exact samples before any timing.
It passed for all frozen filter designs on real EEG (984,480 FP64
values, SHA256 `af4607e6…5b5b`). Both pair orders matched all 578
predictions and work counters.

| Pair order | Original C++ | Vectorized C++ | Replay saving |
| --- | ---: | ---: | ---: |
| Original → vectorized | 110.894 s | 103.802 s | 6.395% |
| Vectorized → original | 110.871 s | 104.488 s | 5.756% |

At a fixed hour and the same replay mix, this increases throughput
**6.11–6.83% relative to the previous C++ build**. Job 3122422 compares
the exact relinked library directly against the frozen solver. Job
3122423 checks the two long forced-adaptation lengths with graph
optimization disabled in both arms.
The exact tested filter source is in the C++ research baseline and
passes local C++ syntax checking. Galvani results were collected after
the connection was restored.

## Vectorized C++ versus frozen

Job 3122422 replayed the same three complete sessions in both run orders
on one A100. All 578 labels in each pair, activity counters, and stage
error guards matched. The process wall includes public data loading and
model loading.

| Pair order | Frozen replay | C++ replay | Time saved | Frozen wall | C++ wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen → C++ | 105.887 s | 104.013 s | 1.770% | 110.982 s | 108.845 s |
| C++ → frozen | 106.910 s | 104.585 s | 2.175% | 111.999 s | 109.494 s |

The three-session ratios correspond to 64–78 seconds saved if frozen
spent an hour on that same solver workload. All sessions recognized at
index 10. Job 3122513 ran all 14 shuffled sessions on one A100 and
matched all 2,655 labels and work counters. All 14 again recognized
at index 10. The larger matched run did not save time.

| Default worker measurement, all 14 sessions | Frozen | Vectorized C++ |
| --- | ---: | ---: |
| Solver replay | 411.785 s | 413.013 s |
| Load model | 4.636 s | 3.988 s |
| Wall since public data loading began | 416.837 s | 417.425 s |

The C++ replay took 0.298% longer, about 11 seconds per hour if the
same replay mix filled the hour. The comparison excludes sealed-worker
overhead and demonstrates zero added evaluation budget.

Job 3122423 ran both arms with deterministic CUDA and unoptimized
TorchScript. The labels match at 378 windows, with all adaptation
counters exact. The 800-window replay differs at indices 405 and 776,
also with all counters exact. Frozen and C++ ran respectively in
147.901 and 142.194 seconds for the 378-window stream and 170.430 and
172.764 seconds for the 800-window stream. The long parity guard failed.
Job 3122514 reproduces this exact pair with intermediate scores to
identify the first stage where the disagreements become visible.

The trace reproduced both arms' 1,178 emitted labels. The maximum
absolute difference was below `3.25e-8` through the inner decoder,
0.00565 at the outer mixture, 0.01832 after the count filter, and
0.25257 in the final personalization scores. All earlier stage
decisions agree at indices 405 and 776. The first outer-mixture error
above `1e-6` occurs at cold window 61. Job 3122538 traced the first
five fine-tuning refits. Class targets matched exactly and FP32
confidence differences were at most `6e-8`.

Job 3122574 changed only the frozen diagnostic's GPU Adam dispatch
from the default multi-tensor path to scalar tensor calls. All 150
labels and work counters stayed exact. The maximum outer-mixture
error versus normal C++ fell from `4.73e-4` to `3.44e-4`, a 27% drop.

Job 3122576 captured the actual fine-tuning input arrays at refits
20 and 60. The raw filter-bank path was bitwise exact. Three other
paths shared tiny differences, up to 3 of 1,180,800 values per slot at
refit 60, with max absolute difference `5.96e-8`. The replays finished.
The initial collector expected two slots. A corrected collector
verified all four using the same logs and arrays.

Job 3122583 gave the C++ refits the exact frozen causal prefixes and
confidences for the first five refits. With the Python comparison
also using scalar Adam, the 70-window outer-mixture max error fell
from `1.157e-6` to `2.168e-8`, close to the input ensemble's error.
All 70 labels matched the originals. The paired controls show that
the optimizer dispatch and tiny input differences both contribute.
The private diagnostic did not change either production solver, and
ordinary long-stream parity remains unresolved.
