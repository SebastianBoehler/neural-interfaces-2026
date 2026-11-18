# Penultimate Track 2 selection · working brief for Wednesday 18 November

**Provisional choice at 04:57 CET: v6.6cap250.** Review at 08:00 and finalize the recommendation by 11:30 CET. Selection cutoff is noon. Friday evening remains the final decision window. The candidate is staged locally.

| Artifact or control | Verified identity / result |
|---|---|
| Candidate archive | `runs/sealed/final_zips/v66_cp1_cap250.zip` · SHA-256 `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` |
| Candidate source | In-archive `submission.py` SHA-256 `37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f` |
| Official competition source | `a66bc23d1a0e686b895c093a642b45c74970758b` |
| Previous sealed archive | v5.1b, SHA-256 `4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef`. Sealed balanced accuracy 81.9767%, wall time 1,287 s. |

**Evidence for selection.** The exact candidate and incumbent were compared on nine external Scherer participants. Cap250 improved from 47.963% to 48.456%, a paired +0.493 pp [80% −0.311,+1.286]. Both orders favored cap250. Population, task timing and channel reconstruction limit transfer. Public new-person dev gives the JS alternative +0.272 pp over cap250. Its conditional returning comparison gives −0.288 pp with only two people. At the campaign's assumed 57/43 mix, the expected difference is about +0.03 pp, too uncertain to displace the already-qualified package. [Exact archive comparison](lunch_qualification385.md) · [Returning-person comparison](js321_returning.md)

**Runtime.** On an A100 the 250-window adaptation cap activated naturally on 320- and 813-window external streams. Reversing the order moved a 52–58 s incremental cost to whichever stream first activated fine-tuning. All decisions and work counts repeated. The private session mix and process layout determine how often this cost recurs. The 60-minute sealed limit remains unverified. [Paired worker timing](runtime387.md)

**How to use the official result.** Check live phase and remaining quota before any upload, and obtain Sebastian's explicit authorization. Record upload ID, actual file hash, score and elapsed time. A higher score validates the aggregate package change on the private cohort. A flat or lower score provides evidence against assuming the public gains transfer and calls for revisiting the returning/new-person mix. A timeout calls for investigating worker startup frequency and full runtime. One aggregate score cannot attribute the change to personalization, fine-tuning or their interaction. Base Friday's choice on the official result and paired evidence available at that point.
