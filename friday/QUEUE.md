# Friday Track 2 queue

Branch `codex/friday-clean-stack`. Active through Friday 20 November, 18:00 CET. The existing Track 2 Friday experiment queue heartbeat is active at 50-minute intervals in this chat. Prefer stronger held-person and external evidence to break alternate ties inside the development noise band.

## Wednesday 18 November, 13:39 CET

| Lane | Scheduler evidence | Next dependency |
| --- | --- | --- |
| F1/F2 waveform, Ferranti | 509981 completed 0:0 in 9 s, with both 4-step source smokes. Arrays 509982 and 509983 accepted. 27 checkpoint/receipt pairs relayed, including the two smokes. | Relay remaining pairs with `bash friday/wave/relay.sh`. Ferranti manifest SHA-256 `58e6e23425f0ee195a62916d130dd406c7a56b75aa77b3b35323846755af8fc2`. |
| F1/F2 held-person, Galvani | 3127060 completed 0:0 in 91 s. Prototype smoke 3127061 and 460 scorer 3127062 released. Centered smoke 3127064 released. Scorers 3127063/65/66 and comparisons 3127067/68 have dependencies. | Validate ten checkpoint pairs per kind/seed before releasing. The Galvani work directory holds `jobs.score.tsv`. Ownership remains with the scorer until it returns completion. |
| F-C1 cap250, Galvani | 3127070 running on its verified source snapshot. | Causal post-latch prior smoke, then eight historical order pairs and returning comparison. |
| F-C2 cap250, Galvani | Registered. | Smoke the actual CSP refit estimator and queue the matched order pairs. |
| Original §458, Galvani | 3127016–18 completed, both P04/P10 timing reports and the blind held-person score verified in the owning release. | Use [the original evidence](../../neural-interfaces-2026/evidence/next_wave_16nov/native458.md). |
| Original §459, Galvani | Real smoke 3127026 completed. External worker array 3127027 was partially complete with continuing tasks at this check. | Read the owning snapshot `ni26_external459_v2_20261118` and official collector before using external scores. |

Both original frozen archives and their SHA-256 manifest are preserved. Friday local package smoke and historical v3 replay reported OK. [Experiment ledger](../docs/LEDGER.md)

## Wednesday 18 November, 13:45 CET

- F1/F2 Ferranti finished both 20-fit arrays and all 42 checkpoint pairs were relayed. The first Galvani held smoke rejected a NumPy string stored as donor metadata under the weights-only loader. No held labels were read. Scope-limited recovery preflight 3127106, smokes 3127114/17, seed scorers 3127115/16/18/19, and contrasts 3127120/21 are in `ni26_friday_wave_20261118v2`. All checkpoints were authenticated unchanged.
- F3/F4 real EEG CPU tests passed. Ferranti 510022 ran the F3 smoke when observed. Arrays 510023/24 and F4 smoke 510025 followed it, then F4 arrays 510026/27. Actor manifest `8e3c09dc6545886a1e9b89c5030fc266484b668d9e981b490dab6fa970274af1`. Relay with `bash friday/wave/relay.sh time`. Galvani release `ni26_friday_wave_time_20261118`.
- Cap250 F-C1 finished 3127070/3127088/3127089. Held-new −0.110 pp and true-returning −0.098 pp, both intervals crossing zero. [Report](cap250-c1-result.md)
- Cap250 F-C2 smoke 3127102 accepted. F-C3 registered for clean-worker external evidence about neural refits.
- F3/F4 Galvani 3127132 preflight, four kind-specific smokes and eight seed scores 3127150–61, and matched comparisons 3127162–67 accepted. F3 smoke checkpoint receipts transferred. F-C2 smoke completed, paired array 3127133 and collector 3127134 accepted. Original §459 completed. Four-order external seed 454 −0.212 pp [80% −0.625,+0.193], seed 1454 0.000 [−0.291,+0.288] vs 48.586% cap250. [Original shared decision handoff](../../neural-interfaces-2026/evidence/next_wave_16nov/EVENING_HANDOFF_18NOV.md)
