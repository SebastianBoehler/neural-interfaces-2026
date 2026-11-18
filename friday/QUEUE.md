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
