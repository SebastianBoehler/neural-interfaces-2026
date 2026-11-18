# Do the exact archives respond selectively to two cognitive tasks?

Cap250 increased the word-specific response on Shin2018 by **5.32 percentage points** [80% paired people interval +3.46,+7.12]. The subtraction-specific response on Shin2017B changed by **+0.52 points** [−0.86,+1.95]. The registered prediction of a resolved improvement on both instruments remains unsupported.

Each participant performed a cognitive task and a matched baseline. For the word study, count how often the package says WORD during word generation versus fixation. Subtract the corresponding change in how often it says SUB. Reverse the two classes for the subtraction study. This measures the specificity of emitted labels to active trials. There is no PROTEUS ground-truth label for fixation, so these differences cannot be added to leaderboard accuracy.

| External population | v5.1b selectivity (pp) | cap250 selectivity (pp) | Paired delta (pp) |
|---|---:|---:|---:|
| 29 people, subtraction versus rest | 4.77 | 5.29 | +0.52 [−0.86,+1.95] |
| 26 people, word generation versus fixation | 14.81 | 20.13 | +5.32 [+3.46,+7.12] |

For word generation, cap250 increased WORD responses on active trials from 37.82% to 38.78% and reduced them on fixation from 25.83% to 24.29%. It also shifted SUB responses in the opposite direction. Separate run, person and order pairing preserves the distinction between the two independently recorded tasks. Only the first four-second crop of each trial enters the worker. Its original 41-channel exports include six native measurement channels for Shin2017B and sixteen for Shin2018. These corpora have previously informed project research. The two studies differ in montage, task, participants and session length.

Both exact archives ran in the official A100 image. The smoke reproduced seven-window reset prefixes and the 31-label archived cap250 public control. The scorer independently reconstructed crop rows and both delivery orders and ran the project replay. No external session triggered recognition. Cap250 made 208 refits and 4,160 optimizer steps across the 52 word-generation streams. Its 174 subtraction streams ended at 20 windows and made zero refits. The main comparison cannot attribute the difference to fine-tuning.

Smoke 3124447, twelve producer shards 3124448, scorer 3124467. [Per-person results and class response rates](selectivity415_result.json) · [Cohort and task provenance](external_semantics351.md) · [Preregistered worker](../../studies/18nov/selectivity415/worker415.py)

## Does longer session memory affect the word response?

The prespecified companion added a reset before windows 21 and 41 in each 60-window word-generation stream. Rows, ordering, people and archives stayed paired. Cap250's advantage over v5.1b was +5.32 pp with the natural 60-window session and +2.18 pp with repeated 20-window resets. The paired horizon effect was **+3.14 pp** [80% +1.28,+5.00]. All 2,080 first-block decisions matched the natural-session controls, and the shortened blocks performed zero neural refits. All seven external EEG archives also matched the earlier §393 input manifest after collection. Both packages maintain multiple types of session state. The experiment identifies a benefit of retaining the longer history on this instrument; it does not isolate a specific algorithm or a leaderboard gain. Smoke 3124547, array 3124548 and scorer 3124549 passed. [Paired horizon results](selectivity415b_result.json)
