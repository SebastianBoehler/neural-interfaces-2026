# Wang subtraction transfer

Wang et al. recorded adults performing three sustained cognitive tasks with eyes closed. They silently subtracted sevens from 5,000, recalled autobiographical events, or sang song lyrics mentally. The task blocks lasted five minutes. Sessions 1 and 2 were about 90 minutes apart; session 3 occurred about one month later. The released dataset identifies 40 of 41 PROTEUS leads by name, with FCz as the acquisition reference. The authors harmonized their original caps for release, and the BIDS files cannot establish direct physical measurement of every lead for every source person. [Original protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC9470564/) · [Version 1.0.1 source](https://openneuro.org/datasets/ds004148/versions/1.0.1)

This source had appeared in the project’s metadata inventory. No prior Wang fit was found in the ledger. Our model learns exclusively on explicit Wang source subjects and recording conditions. Held Scherer/PROTEUS people have no role in fitting. The two target instruments were inspected in previous studies, and inference conditions differ from sustained eyes-closed tasks.

Source people 02–07 each have subtraction as their last cognitive block in session 1, middle in session 2, and first in session 3. The 162 four-second samples are matched across tasks at 30, 150 and 270 seconds. The study fetches only these specific ranges of the tagged public CC0 data, verifying release file MD5 against S3 ETag, object version, HTTP 206 range bounds, BrainVision encoding, channel order, BIDS units and absence of nonsynchronization markers within each selected slice. The data release identifies the cognitive-block order in the participant metadata. [Release license](https://github.com/OpenNeuroDatasets/ds004148/blob/1.0.1/dataset_description.json) · [Block-order fields](https://github.com/OpenNeuroDatasets/ds004148/blob/1.0.1/participants.json)

Arithmetic supervision compares subtraction with silent lyrics and autobiographical memory at matched elapsed times in the same person and session. The three comparison directions are memory versus lyrics, late versus early samples pooled across tasks, and last versus first cognitive block irrespective of task. Exact participant-label sign flips move all sessions and windows of a person together. Source person-held validation includes a stricter session-1 to session-3 test where subtraction changes from last to first. Every target score ranks WORD above SUB using the negative of the source arithmetic direction; no target labels calibrate a threshold. Both four-second FFTs use the same 0.25 Hz bins and six relative bipolar power bands, evaluated at the actual 500 Hz source and 120 Hz target sampling rates.

## Results

The new Wang source improved PROTEUS ranking when all 40 shared leads were available. Its advantage did not reproduce on Scherer, and neither target passed the preregistered donor-sign control. No decoder or upload qualifies from this experiment.

| Equal-person within-run WORD>SUB AUC (%) | PROTEUS, 40 leads | Scherer, 18 leads |
|---|---:|---:|
| Wang arithmetic direction | 57.81 [52.85, 62.62] | 48.47 [45.08, 52.02] |
| Same source and target trials, four shared leads | 44.35 [41.55, 47.29] | 52.01 [48.83, 55.18] |
| Wide minus four, pp | +13.46 [7.79, 19.00] | −3.54 [−5.92, −1.17] |
| Memory versus lyrics control | 43.27 | 47.77 |
| Late versus early control | 45.51 | 52.82 |
| Last versus first cognitive-block control | 41.12 | 53.09 |
| Whole-source-person sign null, 90th percentile | 58.51 | 54.67 |
| Sign patterns at least as high as true result (64 including true) | 12 | 43 |

On PROTEUS, arithmetic exceeded the three actual comparison directions by +14.55, +12.31 and +16.70 pp, respectively. Their separate paired 80% intervals all exclude zero. On Scherer none excluded zero. Target intervals resample people conditional on the six-person source fit; the exact sign-flip distribution varies which source people contribute genuine or reversed class contrasts. These two uncertainty checks answer different questions. [Primary results](wang399_result.json)

**The order reserve found a sign reversal.** Wang sessions 1/2/3 used separately produced PROTEUS AUC 42.94/56.26/57.35%. Session 1 minus 3 was −14.41 pp [−24.83, −3.61], exceeding the registered absolute 5-point drift prior. Scherer gave 54.27/47.17/49.54%, with S1−S3 +4.73 [−2.69,+11.84]. These source sessions change acquisition time and block order together; the experiment cannot attribute the reversal uniquely to either one. Within Wang, other-person arithmetic discrimination using all sessions reached 61.11% on the 40-lead representation and 63.27% on the 18-lead representation. Training on other people in session 1 and evaluating session 3 reduced these to 52.78% and 51.85%. [Reserve results](wang399_order.json)

The 162 distinct windows covered 54 versioned source EEG files and exactly 79,056,000 unique bytes. Four initial source workers stopped on a case-only third-session coordinate mismatch (BrainVision Cpz versus BIDS CPz at the identical index). Two successful subjects were preserved. The repaired reader accepts just that equivalent spelling, verifies all channels and units, and caches successful range digests. Its registered allowance covered at most 36MB of repeated partial transfer. Final sources are in ni26_wang399_v2_20261118, manifest SHA-256 42765fa86b44ad5af1048253c03f341883d9584bf7a9a820b735cc28a47f8b8c. Jobs 3123977/3123978/3123979/3123980/3123981 and retained readers 3123960_3/5 completed. Target score bank digests: PROTEUS 9191a6ccf006d351fd0ff28ecd4df131fcd5e026a40b02f80719a7769ee8d3bc, Scherer 3f9aa9f1340787cf8666b7e6a046e97c7ccf52ec1cb8e05c588eed9e8944fb69.

The six subject-level range-proof files are pinned by SHA-256 manifest d391e33e8ebf554e33d47fc4bf44f292e3d5c239def72113d1ed6aad143a4da2. Final primary/reserve result SHA-256 values are 7066517f6547eca60077acd7460d90c681dfb54614a073f656a6cbd63363d676 and 5e38be300a7d3b87ae81cd0b55531c6c463751b2a7feb8a33250ab2f20a9b97d.
