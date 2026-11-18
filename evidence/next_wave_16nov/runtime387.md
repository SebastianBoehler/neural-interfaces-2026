# Exact-archive long-session runtime

The longest genuine exported PROTEUS session is P10 day 3 with 199 windows. All 14 public sessions were naturally recognized in the §366 worker. Prior cap250 checks passed on forced 378/800-window permutations spanning public recordings. §385 established naturally active adaptation on external sessions up to 120 windows. This check measures the remaining case of a naturally routed continuous recording longer than the 250-window cap. [Prior worker](native366_worker.md) · [Prior external comparison](lunch_qualification385.md)

Two 41-channel external recordings from Schirrmeister2017 contribute 320 and 813 windows, respectively. Each is the complete MOABB 0train recording of one person, without the separate 160-window 1test recording. All 41 target channels are directly recorded. The source tasks are executed movements and rest. This corpus contributed to external pretraining. The workers do not read any task labels. The workload validates execution on these recordings. [Original authors](https://arxiv.org/abs/1703.05051) · [MOABB data loader](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Schirrmeister2017.html) · [Training input manifest](../../scripts/sealed_r19e_inputs.sha256)

Both unmodified archives use the same shuffled rows on the same A100 allocation. Two allocations reverse arm order. Each clean process loads one archive and warms on 31 recognized public P01 windows. The timed interval includes the official fresh-model copy and all B1 calls, with CUDA synchronization. Startup and warm-up are excluded. P10 day 3 uses §366's shuffled order. The two external recordings use independent [387,1] and [387,2] permutation seeds. Only the relative durations are scaled against the historical 1,287-second sealed incumbent.

| Input | SHA-256 |
|---|---|
| Uploaded v5.1b | 4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef |
| Staged cap250 | 9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed |
| v5.1b submission.py | 0259d1eabaf747ac85b178f01832e5f37ea25bddebbebb865d04ec321f498534 |
| cap250 submission.py | 37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f |
| Worker SIF | 7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0 |
| PROTEUS export | 2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327 |
| Schirrmeister export | f65faeacee1fb87627855fbf43938b42dcc4bba5f0d60074348d34e7674ed80e |
| Frozen study source manifest | 1300d341965299abe21a0cd79b9ae4799dde32deca0fa9eeb659b4a389c856e3 |

Galvani directory ni26_runtime387_v1_20261118. Official submodule a66bc23d1a0e686b895c093a642b45c74970758b. Jobs 3123488 (smoke), 3123489 (two paired workers), 3123490 (comparison).

## Original order

| Recording | Windows | Natural route | v5.1b seconds AB / BA | cap250 seconds AB / BA | cap250/v5.1b AB / BA |
|---|---:|---|---:|---:|---:|
| PROTEUS P10 day 3 | 199 | Recognized at 10 | 23.23 / 23.30 | 29.86 / 30.24 | 1.286 / 1.298 |
| Schirrmeister 1, first external recording | 320 | Unrecognized | 36.05 / 36.36 | 125.90 / 128.18 | 3.493 / 3.525 |
| Schirrmeister 2, second external recording | 813 | Unrecognized | 95.77 / 97.16 | 161.77 / 160.85 | 1.689 / 1.656 |

The 320-window ratio exceeded the registered 2.75× near-timeout threshold in both repeats. The first and second unknown recordings each made 24 neural refits and 480 optimizer steps. The 250-window policy skipped six and 56 scheduled refits, respectively. Personalization ran 15 and 40 refits and fused 230 and 723 outputs. No time guard bound or stage logged an error. Peak live/reserved CUDA memory was 4.182/5.543 GiB in both active streams. Both arms repeated all 1,332 outputs and all work counters exactly and cap250 reproduced every archived P10 label. The paired runs used distinct allocations on the same cn222 A100 host. [Detailed results](runtime387_result.json)

Fine-tuning ran for about a minute longer on the first unrecognized recording than on the second despite identical optimizer-step counts. In the original order the first unknown recording was always the shorter one. Multiplying the historical 1,287-second incumbent by either 320-window ratio gives about 75 minutes, but doing so applies the first unknown recording's cost throughout an entire worker run. §387b reverses the two external recordings to separate their order from their lengths. Its source manifest SHA-256 is 07546bd36f55ed71aab07c6731bc4191db1d10121843e5a92ea0b7ee1ef407d9.

## Reversed order and lunch recommendation

Keep cap250 staged. Allow roughly 50–60 seconds of additional cost when the first unknown recording activates fine-tuning in a comparable worker process. Its frequency on the private worker and private total runtime remain unresolved.

In the reversed order, 813 windows took 213.77 s for cap250 versus 94.77 s for v5.1b (2.256×). The 320 windows immediately after took 68.71 s versus 35.59 s (1.931×). Relative to the original-order means, moving the 813 windows first added 52.46 s to cap250. Moving the 320 windows second saved 58.33 s. The incumbent’s corresponding changes were −1.70 and −0.62 s. The two active recordings together took 282.48 s versus the original 288.35 s (−2.04%). All three registered order-effect criteria passed.

Cap250 logged 63 s fine-tuning on its first unknown recording and 12 s on its second, compared with 64–66 s and 13–14 s in the original order. Every recorded label and every model-work count matched the originals for both archives, including all 199 public P10 labels. CUDA peak memory was unchanged. The reversed pair used a different A100 node (cn212), with similar v5.1b durations. This test identifies a first-active-stream cost without isolating any one backend initialization step. The official public evaluator copies a loaded model for each session. The private process layout, mix of recognized people, input lengths and hardware load determine how often this cost occurs and how much of the 60-minute allowance remains. [Reversed results](runtime387b_result.json) · [Official streaming implementation](../../2026-competition/benchmark_utils/streaming.py)

The original three jobs used 19m47s of allocated A100 time. The reversal added 10m38s. All six jobs completed successfully. Original and reversed result SHA-256 digests are 22dd2db65699f4e24e3f91dd09b7863856a1fe80cf6cfdb026cb24346630ed1a and 197576d65b33dfe644c68f13ead08082bb36758db00b248a8d592ce596b7b26a, respectively.
