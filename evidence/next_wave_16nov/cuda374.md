# §374 Coalescing neural posterior transfers

The §370 ordinary-session trace recorded 3,075 kernel launches per window. The
qualified NetBank checked finite values and copied each three-class posterior
to CPU separately. The candidate stacks posteriors on their original device,
copies the bank once and checks the entire bank on CPU. The regular Solver
loads the selected library by absolute package path. This change affects no
model parameters, neural forward or adaptation schedule.
[Baseline build](../../native/worker-parity/README.md) · [Original NetBank](../../native/src/networks.cpp) · [Candidate rebuild](../../studies/17nov/cuda374/build.py)

The experiment uses public shuffled inputs in the verified worker image and
a clean process per arm, with the same 32-window warm-up. Each timed stream
uses a fresh session. The timer includes session copy, all B1 predictions,
host/device synchronization and session destruction. A separate wall measure
also includes loading, warm-up and checks within the process. Stress uses the
existing raised recognition thresholds in both packages. The archived frozen
controls supply expected labels. [Benchmark](../../studies/17nov/cuda374/run.py) · [Schedule and gate](../../studies/17nov/cuda374/run.sbatch)

## Snapshot

All remote files live under Galvani work root
`/mnt/lustre/work/gehler/gfh527/ni26_cuda374_v1_20261117`.
The 2-CPU/16-GiB A100 smoke 3122972 has a 12-minute limit.
Timing 3122973 has a 20-minute limit and requires successful smoke completion.

| Input | SHA-256 or revision |
|---|---|
| Source manifest | `8dee2c2dc85faf7ed103aadcdc94c83c68541553d5f660dc58391057d7e3305a` |
| Link inputs manifest | `3255b49bf3a32a4f623cd565690faf085cefead95abfe1899cc50c49e0712197` |
| Candidate runtime | `7df3d61e3c73d255c9fbef1f31d7a7ee5f56f88c6e6062e51daf5b92e8c3f476` |
| Stock runtime | `4ecfb10ce768fb39165523e34c81a2cd9e4498047f34e7d9c16961445ca8d695` |
| Worker image | `7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0` |
| Public windows | `2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327` |
| Official code | `a66bc23d1a0e686b895c093a642b45c74970758b` |

The numeric prior and promotion rule are registered in [the ledger](../../docs/LEDGER.md).

## Build and smoke

Job 3122972 completed in 278 seconds. Every ordinary 48-window and stress
96-window label matched the freshly run stock arm and archived frozen labels.
All ten work counters matched across implementations in both measured sessions
and their warm-ups. Ordinary recognition occurred at window 10. Stress completed
eight neural refits, 160 optimizer steps, four personalization refits and six
fusions. The synthetic mixed-precision/NaN check passed. The smoke was the
build and behavior prerequisite; timing uses the separate AB/BA allocation.
Galvani smoke.json SHA-256 `444734273e3eeb92db5edcde832e92cfdff6c175e076dc0f38c7233863135df5`.

## Paired warm solver result

Job 3122973 completed in 512 seconds. Every timed label and every activity
counter matched on all eight fresh 96-window streams. Both ordinary pairs
fell below the 2% saving. The active stress result changed sign with order and
the first pair exceeded the allowed 3% slowdown. This candidate does not
qualify for longer confirmation. The result is specific to the two public
prefixes, single GPU and transfer fusion.

| Workload | Order | Stock s | Coalesced s | Saving |
|---|---|---:|---:|---:|
| Ordinary | Stock first | 13.774 | 13.603 | 1.24% |
| Ordinary | Candidate first | 13.697 | 13.678 | 0.14% |
| Active stress | Stock first | 31.705 | 33.186 | −4.67% |
| Active stress | Candidate first | 33.166 | 32.025 | 3.44% |

Galvani timing.json SHA-256 `b5f0f8672fecab55909f5504af9def6d0c5f28ac97ae7ad0c7cdeaae5518640a`.
