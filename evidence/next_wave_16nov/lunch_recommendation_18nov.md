# Penultimate Track 2 selection · working brief for Wednesday 18 November

**Provisional morning choice: v6.6cap250.** The 08:00 review retained this choice. The recommendation is due by 11:30 CET. Selection cutoff is noon. Friday evening remains the final decision window. The candidate is staged locally.

| Artifact or control | Verified identity / result |
|---|---|
| Candidate archive | `runs/sealed/final_zips/v66_cp1_cap250.zip` · SHA-256 `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` |
| Candidate source | In-archive `submission.py` SHA-256 `37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f` |
| Official competition source | `a66bc23d1a0e686b895c093a642b45c74970758b` |
| Previous sealed archive | v5.1b, SHA-256 `4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef`. Sealed balanced accuracy 81.9767%, wall time 1,287 s. |

**Public development evidence.** Historical full-system fold simulations estimate cap250 minus v5.1b at +1.556 pp [80% +1.009,+2.071] on ten new people, with nine improving. Returning P04 and P10 gained +1.808 and +0.682 pp. The planning mix yields +1.274 pp. The simulations differ from the archive in auxiliary ensemble size, input order for some components and returning-route behavior. Both release archives trained on the public data, so their literal public replay cannot measure this held-person effect. [Lineage and paired measurements](lineage419.md)

**Evidence for selection.** The exact candidate and incumbent were compared on nine external Scherer participants. Cap250 improved from 47.963% to 48.456%, a paired +0.493 pp [80% −0.311,+1.286]. A second pair of fresh orders gave +0.040 pp [−0.837,+0.929], with one order favoring v5.1b. The four-order mean is +0.266 pp on the same nine people. Population, task timing and channel reconstruction limit transfer. Public new-person dev gives the JS alternative +0.272 pp over cap250. Its conditional returning comparison gives −0.288 pp with only two people. At the campaign's assumed 57/43 mix, the expected difference is about +0.03 pp, too uncertain to displace the already-qualified package. [Exact archive comparison](lunch_qualification385.md) · [Order replication](cross411b_package.json) · [Returning-person comparison](js321_returning.md)

For JS versus cap250, the break-even share of new people is about 51.4% under the assumed returning-person mix. At 40%, 57% and 70% new, the projected JS differences are −0.064, +0.031 and +0.104 pp respectively. These are sensitivity calculations from two development estimates. The private mix and both underlying effect sizes are uncertain.

The exact candidate increased a word-versus-subtraction label contrast on 26 external word-generation participants by +5.32 pp [80% +3.46,+7.12]. Resetting the same archives every 20 windows reduced that advantage to +2.18 pp. The paired benefit of the longer history was +3.14 pp [+1.28,+5.00]. The subtraction study yielded +0.52 pp [−0.86,+1.95]. Fixation has no competition label, which limits these measurements to task selectivity. [Task and reset controls](selectivity415.md)

Preserving original run blocks on the external Scherer data gave +0.463 pp and shuffling the blocks gave +0.142 pp, versus +0.040 pp under matched full-session shuffling. All three intervals included zero. The possible change in package advantage exceeded the preregistered ±1 pp equivalence margin. [Order-scope comparison](scope416.md)

**Runtime.** On an A100 the 250-window adaptation cap activated naturally on 320- and 813-window external streams. Reversing the order moved a 52–58 s incremental cost to whichever stream first activated fine-tuning. All decisions and work counts repeated. The private session mix and process layout determine how often this cost recurs. The 60-minute sealed limit remains unverified. [Paired worker timing](runtime387.md)

Two clean A100 allocations compared the exact archives on the same four 190-window sessions. Cap250 took 584.05 versus 349.50 seconds on one, and 211.22 versus 114.61 seconds on the other. All 3,040 archived decisions matched. Both standardized workloads fit the historical headroom under the tested process scenarios. The hidden workload and hardware remain unknown. [Exact-archive timing and scenario bounds](budget414.md)

A private count-filter optimization saved 8.43 and 8.39 seconds in two complete A100 worker runs. That is approximately 2.4% of those three-recording runs. The optimization is outside the staged archive. [Worker comparison](qualification408.md)

**Historical transfer.** Two prior package changes give concrete reference points. The v4.1 change gained +0.58 pp on its development measure and +0.267 pp sealed. The v5.1b change gained +0.89 pp on development and +0.361 pp sealed. Both changed returning-person handling. Cap250 also adapts on new people, so multiplying its development or external effect by either historical ratio would give an unsupported forecast. [State and metric history](../../docs/STATE.md) · [Official upload log](../../docs/SEALED_UPLOADS.md)

**How to use the official result.** Check live phase and remaining quota before any upload, and obtain Sebastian's explicit authorization. Record upload ID, actual file hash, score and elapsed time. A higher score validates the aggregate package change on the private cohort. A flat or lower score provides evidence against assuming the public gains transfer and calls for revisiting the returning/new-person mix. A timeout calls for investigating worker startup frequency and full runtime. One aggregate score cannot attribute the change to personalization, fine-tuning or their interaction. Base Friday's choice on the official result and paired evidence available at that point.
