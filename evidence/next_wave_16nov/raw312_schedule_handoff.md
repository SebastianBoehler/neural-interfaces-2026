# §312 timetable handoff — 17 November 2026, about01:32CET

- CPUcomplete T/FBCNet/P01/seed0cell3116416passed:242.21s, actual01:23:55–01:27:57CET, six bound CPUs/cn002, Slurm Python stepRSS1,841,900KB. No held accuracy read.
- Independent all-CPU160cell route: `/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2`; sourceSHA528284dc7c07c835534ca94e6637c00df44a38755b62313d5cfbfac25a265ef4. Same scientific recipe, CPU-native RNG/arithmetic recorded; never mix with GPUfolds.
- CPUtiming gate3116419passed. Wave0array3116421covers0–7, max4concurrent;0–2started01:28:59CET with20min requests. Pending3–7now10min.
- CPUfuture scheduling callback3116430follows3116421, ordinal1. It uses separately immutable`ni26_raw312_cpu_schedule_20261117/wave.py` to request10min and chain all later callbacks. Old pending3116422was cancelled. Frozen scientific source/epochs are unchanged.
- CPUwaves contain8tasks, use up to4six-CPU/24GB allocations, and check≤32running/272submitted before every release. SourceCPUprobe80is canonical; remaining159cells are allCPU. Complete160cell collector is queued only at the last wave; itsIDwill appear in`collector_job.json`.
- CPUprovisional full-matrix window05:00–08:00CET, conditional on sustained slots/similar variant costs/no preemption/passing guards. Ideal2.7h extrapolation is compute-equivalent only. Whole score and chain stages follow afterward.
- GPUcanonicalarray3115990pending requests45→10min. Last forecast first04:20CET, last first80cell14:11CET; volatile, not promised. Scientific source remains original8275d60d…6559.
- GPUwave2scheduling callback3116412follows3115990 and explicitly requests10min for remaining80cells. Original pending3115991cancelled;315bridge3116075now depends afterok3116412.
- Separate5min completeGPU Tcell probe3116413writes separate outputs, not canonical ones. Last forecast04:53CET; no fullGPUproducer cost measured yet. Its first pending launcher3116411was replaced before execution after an export-variable issue.
- Original315consumer remains GPU-pinned. A completeCPUconsumer needs prospective registration/new source pin, chosen from availability/timing rather than outer accuracy. No silent fallback or mixed-device completion.
- CPUsetup3116414failed before training on copied read-only run.py; corrected immutable setup3116415passed. No running source was mutated.
- Root owns current timetable/portfolio/STATE. Re-estimate from CPUwhole-cell completions and`cpu_jobs.jsonl`; do not quote24GPUhours or24CPUallocation-hours as wallclock.
