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
index 312. The matched counts and timing do not qualify equivalent outputs.
Job 3122389 repeats the first 330 inputs twice per arm with explicit
deterministic CUDA settings to isolate sources of repeat variation.

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

Job 3122137 tests whether running the two independent tangent cores
concurrently saves time. It changes only scheduling and uses the same
three complete real sessions, two CPUs, one A100 and AB/BA comparison.
