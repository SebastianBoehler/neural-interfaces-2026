# Friday Track 2 experiments

The two primary research questions in each wave share the same ten public held-out people (P01–P10), 14 sessions, eight shuffled orders and 28 metric cells. The cohort supplies 2,655 windows per order. P04 and P10 are evaluated as new people here. Returning-person qualification requires separate evidence.

| Wave | New learned session memory, §454 | Cap250 refinement, §455n |
|---|---|---|
| First | Learn which past spectral patterns are relevant to the current window. Train with the held person excluded and compare genuine history against a source donor of the same age. | Test the implicit class prior in the existing personalization fit. |
| Second | Learn how spectral features vary together in past windows, and use those relationships to adjust the current prediction. | Test when previous evidence supports trusting the personalization fit. |

The learned corrections apply to the full frozen decoder. They use band powers from each EEG window. The original whole-waveform A6 study remains a distinct experiment. §455a is an auxiliary check on agreement between two trained memory models.

Use both unchanged full-system references. The historical nine-eye replay reproduces 76.2793% and allows direct comparison with §443. The three-eye/three-rest replay reproduces 76.104% and matches the staged auxiliary model counts. Neither cohort supplied an independent test of a method chosen after inspecting these public results. The noon cap250 ZIP remains the submission reference. [Historical baseline](history443.md) · [Staged seed-count replay](aux430.md)

Every comparison uses unchanged predictions on the same deliveries. Report absolute scores, paired differences, uncertainty across people and delivery orders, how many decisions changed, and P04/P10 results separately. A new outcome qualifies as a candidate for further testing after inspecting effect size, repeatability, and the actual complete-system code path. Results and job dependencies are recorded in the [experiment ledger](../../docs/LEDGER.md).

### First dispatch, Wednesday 18 November

Ferranti accepted H100 smoke 509842 and the two 20-fit arrays 509843/509844. The augmentation follow-up tests a consistent gain change within each session against changes reassigned between windows. Smoke 509845 qualifies its two 20-fit arrays 509846/509847. Galvani accepted the full-stack replay jobs 3126893–3126904. A workstation relay copies prediction banks from Ferranti and releases each scorer when the required files arrive. Inputs, actor receipts and source hashes are checked.

On Galvani, the §455n historical full-system smoke, eight-order replay and both collectors (3126852–3126855) finished. Prior correction +0.009 pp [80% −0.014, +0.035]; past agreement weighting −0.042 pp [−0.099, +0.013]. Returning-role smoke, replay and collector 3126889–3126891 test P04/P10. The §456 reconstruction uses the authentic §430 feature branches, count filter and personalization stages to compare against the staged three-seed counts. [Native readout methods and results](native455.md)

A rolling queue check runs every 50 minutes through Friday 20 November, 18:00 CET. It can start new preregistered waves and recover failed dependencies. The split applies to research lanes. Slurm assigns hardware independently, and the two families require different amounts of CPU and GPU work.
