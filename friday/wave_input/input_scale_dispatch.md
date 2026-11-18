# F11 · Source release receipt

The active comparison is causal sensor-power scaling trained on same-session history versus a matched source donor. Both arms use the F7 raw encoder and its supervised objective with the detached masked auxiliary. Their ledgers bind the exact source windows, donor schedule and initial model. [F11 preregistration](../../docs/LEDGER.md#f11--adapt-sensor-scale-before-the-raw-waveform-encoder)

- Ferranti 510567 completed 0:0 in 11 seconds. Both P01 actors passed four optimizer updates, bitwise first-window identity, active gains, mean preservation, causal prefixes, reset and one-window logit equivalence. Fitted reference power, saved donor features, source row sets, target and donor schedules, and initial weights matched exactly across the arms.
- Ferranti arrays 510568 and 510569 cover two seeds by ten held people per arm. Both arrays follow the successful smoke. Early cells 510568_0 and 510568_2 completed, and the rest were waiting for GPU priority at the first relay.
- Galvani independent scorer preflight 3127923, paired smoke 3127924 and complete seed scorers 3127925/26 were accepted. The first verified relay delivered both real P01 smokes. Each complete scorer needs twenty authenticated production checkpoint pairs for its seed.
- The owning directories are `~/ni26/friday_wave_scale_20261118` on Ferranti and `/mnt/lustre/work/gehler/gfh527/ni26_friday_wave_scale_20261118` on Galvani. Source commit `4c0e364`. Actor manifest SHA-256 `501909d4e1623d13bb62e9f780df7d23db34f39003df3a6eba42bf2b04a12fed`.

The source smoke measured training and control activity. Held decisions and package performance remained pending at this receipt.

At 15:57 CET, 39 qualifying production fits had completed in at most 8 seconds. The remaining genuine P02/seed 460 task, 510568_1, was automatically requeued after a node affinity error before Python. It became eligible at 15:48:46 CET. Its pending wall limit was reduced to two minutes, then to one minute at 16:06 for backfill. The model, H100 pool and fold are the same. Galvani 3127926 completed the authenticated seed 1460 comparison. Score 3127925 remains held for the genuine P02 checkpoint pair.
