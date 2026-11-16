# §300 saved-stage attribution

This registered diagnostic asks where the observed complete-arm losses first appear in outputs that were saved. It compares V6Fr after rebalancing/SessionEM/TSEM, the original count filter recomputed from V6Fr, and the saved final CP argmax. It cannot distinguish effects within rebalancing, SessionEM and TSEM. A later repair requires a separately registered intervention.

The collector reuses the complete §294/§292A guards, original count-filter code and original paired accuracy and bootstrap functions. Its independent class/cell-weighted correction and destruction breakdown must sum to the original accuracy delta. It reports fixed confusion and person effects. Brier and true-class margin are computed only for arrays verified to be probabilities; final CP scores are used only via argmax.

Before the full run, smoke job 3114186 instruments the pinned original CP code in memory, without output writes, and checks its filtered control input against the independently recomputed count filter bitwise. The collector repeats this check, pins the §300 ledger snapshot, and records the exact output/source hashes and UTC sequential-read history. Smoke is pending; no §300 diagnostic score has been read. Source: `studies/16nov/followups/attribution/`.

Control-input smoke 3114186 completed 0:0: recomputation matched the pinned original CP input bitwise and both probability stages passed validation. V2 also records the actual loaded count-filter module paths and hashes and verifies each against its nominal pinned copy. Full attribution job 3114212 is queued. The fixed confusion matrix is weighted by cell and class; each row sum is its class share, rather than a separately normalized class recall.
