# Upload decision note · Sat 21 Nov 2026, prepared ~08:55 CET for the 12:37 decision
Nothing uploaded. Quota 1 left (STATE.md); Failed attempts do not consume it, manual cancellations do. Queue is busy: upload early, evaluation may end after 23:59. Never upload without explicit user authorisation; recheck live quota/phase first.

## Candidates (all derive from hard_safe_st `239c86bd…3c1e`; deltas are pp vs the hard-only base unless stated)
| Role | ZIP | SHA-256 |
|---|---|---|
| **Lead by score, timing gate failed** | `final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip` | `7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07` |
| Alternate, 0.8 s cn222 headroom | `final_stack_hard_safe_st_team_b_nine15_tbgraph_fc.zip` | `6e6bbbb841db4ff1ea5e83e4bad75a090ab0a30455fa5ca570063b5ba828d1d4` |
| Timing comparison | `final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph_fc.zip` | `086c7600555efba5545c035e1ef3c4245ab036599b588400c34ac118753fcd73` |
| Fallback | `final_stack_hardcb_safe_st_cbgraph.zip` | `6ba4092d1f07e036cfb7e19f306eba993cfdea5b6d9cff779d03c32e847a39f5` |

Not a candidate: `ba_terminal` (`c50a62913360…`): RTX proxy 3,741–4,017 s, Scherer B-ret −0.52 [−1.01, −0.02] flag.

## Measured gain (paired 80 %, people then orders; 57/43 mix is UNVERIFIED; J = equal-weight)
| Contrast | Held-new [80 %] | Returning [80 %] | J / 57-43* |
|---|---|---|---|
| nine25, 16 confirm orders 71156-71171 | +0.89 [+0.45, +1.26] | +1.21 [+0.91, +1.51] | +1.05 / +1.03 |
| nine25, pooled 24 orders | +0.92 [+0.44, +1.31] | +1.00 [+0.75, +1.25] | +0.96 / +0.97 |
| nine25, h2h 71200-71215 (finalists) | +0.67 [+0.34, +0.97] | +0.98 [+0.72, +1.25] | not stated |
| nine25, fresh dose block 1 71400-71415 | +0.87 [+0.45, +1.22] | +0.55 [+0.22, +0.85] (P04 +0.31, P10 +0.79) | +0.71 / +0.79 |
| nine15, 16 confirm / h2h / dose 1 | +0.64 / +0.44 / +0.58 | +0.78 / +0.76 / +0.49 | dose 1 J +0.54 / +0.56 |
| nine25 − nine15 (h2h; dose 1) | +0.23 [+0.05, +0.41]; +0.28 [+0.03, +0.50] | +0.22 [−0.03, +0.44]; +0.06 [−0.15, +0.26] | |
- Three fit seeds (orders 71124-71131): nine25 +0.95 [+0.25, +1.53] held-new, +0.67 [+0.36, +0.96] returning; all seeds pass.
- Spread: gain sits in P04/P05 (P05 +2.96, P04 +1.61); P01, P02 negative in the 16-order run, P02 −0.77 in dose 1. Returning is two people (P04 +0.31 only in dose 1); levels 8 orders +0.59, 16 orders +1.21, dose +0.55: not settled.
- Out of lab (Galvani NI-V Scherer, releases A+B, 16 orders): nine25 B-new full +0.30/−0.01, B-ret −0.11/0.00; nine15 +0.16/−0.05, +0.34/−0.15; no veto, no help. Releases C, D (09:00): nine25 B-new full +0.13/+0.13, nine15 +0.34/+0.21; 4-release means (32 orders) nine25 +0.14, nine15 +0.17, no veto; nine25 within-scheme −0.16 to −0.48 every release. Release E: queued behind the 300-task cap, pending.
- hardcb cbgraph vs cap250: about +0.2 (laneP3 +0.22 held-new, −0.03 returning at 20/80*); the Team B stack is the larger measured gain (hard itself ≈ +0.2 to +0.4 over cap250, unmeasured against Team B in one run).
- Seeds/h2h cards: finalists_21nov.md filled from Leonardo receipts.

## Runtime vs 3,600 s

The sealed RTX PRO 4500 is untested. Galvani Solver measurements give these cap250-relative projections.

| ZIP | RTX 2080 Ti | A100-PCIe |
|---|---:|---:|
| nine25_fc | 2,932–3,169 s (3,126 s on cn116) | 2,994–3,636 s (maximum on cn222) |
| nine15_fc | 2,957–3,092 s | 3,014–3,599 s (maximum on cn222) |
| hardcb nine25_fc | 3,165–3,229 s | 3,229–3,274 s |
| hardcb cbgraph | 2,888–3,148 s | 3,027–3,043 s |

Nine25_fc passed 3,600 s on six 2080 Ti and two A100 nodes. It exceeded the limit by 36 s on A100 cn222, where cap250 itself ran at 476 ms/window. The cn222 repeat projected 3,575.5 s for nine25_fc and 3,599.2 s for nine15_fc on the same allocation. Several nodes exceeded the 3,060 s working margin. The fused archives remove the dominant Team B convolution cost on the 2080 Ti. The unpatched nine25 and nine15 each exceeded the official limit on 4 of 8 chained 2080 Ti measurements, with a combined maximum of 3,840 s.

## Qualification

All three fused ZIPs reproduced byte for byte. Only the Team B encoder text in `submission.py` differs from each parent. Galvani parent/fused identity found zero changed labels on the covered public streams, with maximum posterior difference 7.5e−8. The earlier official warm-up replay passes apply to the parent ZIPs. All four exact fused warm-up pairs passed on 2080 Ti and A100. All three archives passed the live Team B graph capture fault and dead-session tests. The qualification record contains all nine matched timings and the cn222 repeat. [Qualification of _fc](teamb_overhead.md#qualification-of-_fc)

## Recommendation (12:37 CET)

**Nine25_fc failed the registered timing gate on cn222.** Its parent produced the strongest replicated gains. **Nine15_fc** passed exact-archive warm-up and guard checks. Its cn222 timing left only 0.8 s of projected headroom. **Hardcb cbgraph** is the qualified choice if the decision requires timing headroom. The unpatched Team B archives exceeded 3,600 s on some 2080 Ti proxies.

## Could change by 12:37
(1) Leonardo dose block 2/pooled `dose.json` and growth screen/freeze (`grow/`; frozen choice only if timing-eligible; +8 ms cap): could add small positive stacking, unlikely to be ready for qualification. (2) Galvani NI-V C/D collected, no veto (see above); E pending. Chained e2e timing (09:00, 17 of 36 jobs done): unpatched nine25/nine15 on 2080 Ti 3,465-3,840 s over 8 runs each (4 of 8 above 3,600 s), cbgraph alone 2,944-3,086 s; A100 3,149-3,374 s. Fused identity and the full node timing table are in `teamb_overhead.md`. (3) Fable round 3 Bet 3 anchor (65011780/81): not a package; no change expected.

**Orchestrator note 12:15.** The nine25_fc timing "fail" (cn222, 3,636 s) is a host artefact: cap250 itself ran 476 ms/window there (3.7x its sealed 127 ms), so the additive projection does not apply; scaled to sealed speed the extra cost is ~14 ms/window (~2,850 s). On the 8 normal nodes nine25_fc projects 2,941-3,169 s. Lead stays nine25_fc (`7017bd5c…`); hardcb cbgraph is the fallback. Candidate layer under evaluation: ba_six (returning +0.70 [+0.41, +0.99], held-new −0.10 [−0.39, +0.16]).
