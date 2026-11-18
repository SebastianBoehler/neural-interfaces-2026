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

The next cap250 refinement acts before the late personalization stage. It compares trust in the class-count model inferred from previous predictions against fixed trust and stock. The real-data smoke passed; eight-order array 3126940 and collector 3126943 were submitted. The returning-role recovery uses jobs 3126944–3126946. Both Ferranti H100 qualification jobs passed and production fits began at 13:10 CET. [Ferranti mechanisms and checks](memory454.md)

### Wednesday score and rank reference

At 13:19 CET on 18 November, the public Codabench results for sealed task 36954 put v5.1b at entry #20 with 0.8197668759. The #19, #15, #14, #13 and #10 entries scored 0.8207452902, 0.8237705196, 0.8253647372, 0.8314157695 and 0.8350310945. The #5 entry scored 0.8412138586. A cap250 score of 0.824–0.829 would occupy #15–14 with those entries fixed. This planning scenario predates the upload and has no calibrated probability. [Codabench results](https://www.codabench.org/api/leaderboards/19997/)

On deployment-count-matched public new-person folds, learned retrieval improved by +0.002174 and +0.001583 in 0–1 score units. Assuming the full measured effect on 56.6% new-person cells and zero on returning cells yields +0.001231 and +0.000896 on top of an unknown cap250 sealed score. The fixed early-count change yields +0.000354 under the same assumptions. These are separate comparisons to one stock baseline. [Learned memory results](matched456_learned.json) · [Early count results](early455_result.json)

The clean-checkout campaign records new jobs and the original graph and external receipts in the [Wednesday decision handoff](EVENING_HANDOFF_18NOV.md).
