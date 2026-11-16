# §298 sequential complete-arm reads

§298 in ledger commit `eed7c0e` allows a registered arm to be read only after all eight orders and all six base variants, all zero-order/CP controls, pinned provenance and gate checks. All original arms remain required in the final report. The same repeatedly reused ten dev people make sequential and selected results exploratory.

The system and gated collectors use the existing guarded scorer and metric functions. Each records UTC time, exact SHA-256 hashes of the outputs and controls read, the registration hash, and earlier blocks in the shared locked Galvani journal `/mnt/lustre/work/gehler/gfh527/ni26_rolling_20261116/reads.jsonl`. The gated collector additionally verifies eight true ZG06 zero orders and reports each base's mask counts.

The first two collector jobs failed before any score: isolated Python omitted the helper directory. A new immutable v2 snapshot passed imports under the actual Galvani `python -B -I`; the failed snapshot remains available for audit. All seven §294 zero jobs completed 0:0. Serialized replacement collectors 3114111–3114118 cover E05CS, E1CS, E2CS, D05CS, E05, D05, FBL15 and EFBL. The first is running; the rest wait on the preceding collector. EG06 collector 3114119 also waits on gated-zero array 3114085. No early score has yet been reported.
