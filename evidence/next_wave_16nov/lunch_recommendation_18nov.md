# Penultimate Track 2 selection · working brief for Wednesday 18 November

**Recommendation for the penultimate submission: v6.6cap250.** At 10:54 CET the staged ZIP and in-archive source reproduced the hashes below and the ZIP integrity check passed. Selection cutoff is 12:00 CET. Friday evening remains the final decision window. The candidate is staged locally and awaits upload authorization.

| Artifact or control | Verified identity / result |
|---|---|
| Candidate archive | `runs/sealed/final_zips/v66_cp1_cap250.zip` · SHA-256 `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` |
| Candidate source | In-archive `submission.py` SHA-256 `37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f` |
| Official competition source | `a66bc23d1a0e686b895c093a642b45c74970758b` |
| Previous sealed archive | v5.1b, SHA-256 `4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef`. Sealed balanced accuracy 81.9767%, wall time 1,287 s. |

**What ships.** The Python submission takes one EEG window, uses CUDA for its neural components, and returns one of three labels immediately. It runs the v5.1b ensemble and routing. On unrecognized sessions cap250 can fine-tune from the delivered prefix and personalize its predictions; a recognized person uses the returning route. Session state resets at the next session. The A6 waveform-to-label model and the C++/CUDA port are separate research candidates. [Package build](../../docs/V6_BUILD.md) · [Whole-model comparison](joint382.md) · [Native worker](native366_worker.md)

**Public development evidence.** Matching the staged three eye and three rest seeds in the historical new-person simulation gives 74.724% for v5.1b and 76.104% for cap250. The paired gain is +1.380 pp [80% +0.742,+1.988], with nine of ten people improving across eight orders. The original simulation gave +1.556 pp. Matching the deployed seed counts on returning P04 and P10 gave +1.483 and +0.575 pp. The corrected public planning mix gives +1.110 pp. [Returning correction](aux431.md) Some component orders and returning-route details still differ from the staged archives. [Auxiliary correction](aux430.md) · [Historical measurements](lineage419.md)

On Scherer, changing how 22 missing channels were estimated moved the two-order package difference from +0.040 to −1.088 pp, with all 19 recorded channels identical. On the other order pair the same change moved +0.493 to −0.823 pp, a paired deterioration of −1.316 pp [−2.216,−0.400]. [Order replication](montage446_result.json) Moving the crop 250 ms later on the separate §385 order pair moved the difference from +0.493 to −0.390 pp. Both results limit the use of external package scores for a sealed forecast. [Channel reconstruction](montage428.md) · [Crop replication](cue425b_result.json)

**Evidence for selection.** The exact candidate and incumbent were compared on nine external Scherer participants. Cap250 improved from 47.963% to 48.456%, a paired +0.493 pp [80% −0.311,+1.286]. A second pair of fresh orders gave +0.040 pp [−0.837,+0.929], with one order favoring v5.1b. The four-order mean is +0.266 pp on the same nine people. Population, task timing and channel reconstruction limit transfer. Public new-person dev gives the JS alternative +0.272 pp over cap250. Its conditional returning comparison gives −0.288 pp with only two people. At the campaign's assumed 57/43 mix, the expected difference is about +0.03 pp, too uncertain to displace the already-qualified package. [Exact archive comparison](lunch_qualification385.md) · [Order replication](cross411b_package.json) · [Returning-person comparison](js321_returning.md)

The protected JS policy kept a separate cap250 history for use after recognition. It reduced the conditional returning cost to −0.087 pp. On nine external Scherer people it scored 47.900% versus 48.456% for cap250. Both orders favored cap250. All sessions stayed on the unrecognized route, so a recognition error cannot explain this external decline. [Protected policy](protected436.md)

For JS versus cap250, the break-even share of new people is about 51.4% under the assumed returning-person mix. At 40%, 57% and 70% new, the projected JS differences are −0.064, +0.031 and +0.104 pp respectively. These are sensitivity calculations from two development estimates. The private mix and both underlying effect sizes are uncertain.

The public emulation still favored cap250 by +0.349 pp in the least favorable delivery scheme after omitting any two people. On the nine external Scherer people, omitting one person reversed the exact-archive advantage to −0.346 pp. The historical public result is more robust to participant choice than the external package comparison. Choosing JS or cap250 with each evaluation person excluded gave +0.032 pp versus always using cap250 [80% −0.253,+0.130]. [Release sensitivity](release420.md)

The exact candidate increased a word-versus-subtraction label contrast on 26 external word-generation participants by +5.32 pp [80% +3.46,+7.12]. Resetting the same archives every 20 windows reduced that advantage to +2.18 pp. The paired benefit of the longer history was +3.14 pp [+1.28,+5.00]. The subtraction study yielded +0.52 pp [−0.86,+1.95]. Fixation has no competition label. Recorded eye signals alone distinguished the task on 13 reserve participants at 58.59% accuracy, and matching eye activity gave inconclusive attenuation. Task selectivity provides no measure of cortical specificity or sealed accuracy. [Task and reset controls](selectivity415.md) · [Eye-signal controls](ocular421.md)

Preserving original run blocks on the external Scherer data gave +0.463 pp and shuffling the blocks gave +0.142 pp, versus +0.040 pp under matched full-session shuffling. All three intervals included zero. The possible change in package advantage exceeded the preregistered ±1 pp equivalence margin. [Order-scope comparison](scope416.md)

On held public people the historical gain covered all three classes. Motor imagery gained 2.00 pp, subtraction 1.67 pp and word generation 1.00 pp. Later windows accounted for more gain than their share on public data. That pattern weakened under full shuffle and did not reproduce clearly on Scherer. [Paired class and delivery analysis](timeline427.md)

Moving the Scherer window start from 1.00 to 1.25 seconds after cue onset removed direct overlap with the displayed cue. The package difference changed by +0.017 pp [80% −0.797,+0.813]. At 1.50 seconds v5.1b led by 0.585 pp on those two orders. On the separate Shin2018 word task, aligned eye-signal subtraction changed 22–24% of individual archive decisions. Its effect on cap250’s relative word selectivity remained unresolved. [Cue timing](cue425.md) · [Measured eye-signal intervention](eye426.md)

A session-specific neural correction added +0.021 and +0.002 pp to the full historical decoder across two initializations, with both 80% intervals crossing zero. Blending 5% of the entire model lowered accuracy by 0.311 and 0.242 pp. A separate motor-only insertion lowered accuracy by 0.374 pp [−0.546,−0.132]. [Learned memory](history443.md) · [Motor imagery](motor440.md)

**Runtime.** On an A100 the 250-window adaptation cap activated naturally on 320- and 813-window external streams. Reversing the order moved a 52–58 s incremental cost to whichever stream first activated fine-tuning. All decisions and work counts repeated. The private session mix and process layout determine how often this cost recurs. The 60-minute sealed limit remains unverified. [Paired worker timing](runtime387.md)

Two clean A100 allocations compared the exact archives on the same four 190-window sessions. Cap250 took 584.05 versus 349.50 seconds on one, and 211.22 versus 114.61 seconds on the other. All 3,040 archived decisions matched. Both standardized workloads fit the historical headroom under the tested process scenarios. The hidden workload and hardware remain unknown. [Exact-archive timing and scenario bounds](budget414.md)

A private count-filter optimization saved 8.43 and 8.39 seconds in two complete A100 worker runs. That is approximately 2.4% of those three-recording runs. The optimization is outside the staged archive. [Worker comparison](qualification408.md)

**Historical transfer.** Two prior package changes give concrete reference points. The v4.1 change gained +0.58 pp on its development measure and +0.267 pp sealed. The v5.1b change gained +0.89 pp on development and +0.361 pp sealed. Both changed returning-person handling. Cap250 also adapts on new people, so multiplying its development or external effect by either historical ratio would give an unsupported forecast. [State and metric history](../../docs/STATE.md) · [Official upload log](../../docs/SEALED_UPLOADS.md)

**How Wednesday informs Friday.** Before any upload, confirm live phase and quota and obtain Sebastian's explicit authorization for this archive. Record its ID, ZIP hash, individual submission score and elapsed seconds. Use the individual score even if the best-score leaderboard continues to display v5.1b. Compute the change in percentage points as `100 × (new score − 0.8197668759)`. For example, a score of 0.82500 would be **+0.5233 pp**. The public development gain cannot be added to the sealed score directly.

| Wednesday outcome | Friday implication |
|---|---|
| Cap250 beats 81.9767% and finishes within budget | Use cap250 as the private reference for any finalist. Require paired evidence for the *additional* Friday change, including its cost. |
| Cap250 ties or trails v5.1b | Keep v5.1b as the sealed reference. Reassess release changes under both returning- and new-person assumptions; the aggregate score cannot identify which component lost. |
| Cap250 times out or fails | Use the actual job status and available diagnostics to select the matching worker replay. Recheck quota before planning a replacement. |
| No Wednesday run | Friday has no new private observation. Compare candidates with the measured public, external and runtime evidence and retain this uncertainty. |

Friday evening, 20 November is the final decision window.
