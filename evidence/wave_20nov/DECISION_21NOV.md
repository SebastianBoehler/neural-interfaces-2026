# Upload decision note · Sat 21 Nov 2026, prepared ~08:55 CET for the 12:37 decision
Nothing uploaded. Quota 1 left (STATE.md); Failed attempts do not consume it, manual cancellations do. Queue is busy: upload early, evaluation may end after 23:59. Never upload without explicit user authorisation; recheck live quota/phase first.

## Candidates (all derive from hard_safe_st `239c86bd…3c1e`; deltas are pp vs the hard-only base unless stated)
| Role | ZIP | SHA-256 |
|---|---|---|
| **Lead** | `final_stack_hard_safe_st_team_b_nine25_tbgraph.zip` | `6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9` |
| Alt | `…_team_b_nine15_tbgraph.zip` | `9ad45b11de1bcc356c56a0b7ff9ae74eb0c471f64546cb6a18f716b2f956188b` |
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
- Out of lab (Galvani NI-V Scherer, releases A+B, 16 orders): nine25 B-new full +0.30/−0.01, B-ret −0.11/0.00; nine15 +0.16/−0.05, +0.34/−0.15; no veto, no help. Releases C, D, E: **pending (Galvani down)**.
- hardcb cbgraph vs cap250: about +0.2 (laneP3 +0.22 held-new, −0.03 returning at 20/80*); the Team B stack is the larger measured gain (hard itself ≈ +0.2 to +0.4 over cap250, unmeasured against Team B in one run).
- Seeds/h2h cards: finalists_21nov.md filled from Leonardo receipts.

## Runtime vs 3,600 s (sealed GPU is an RTX PRO 4500, untested; hosts differ; Team B cost is launch/CPU bound)
| ZIP | RTX 2080 Ti (Galvani) | A100-PCIe |
|---|---|---|
| nine25 | 3,545-3,657 s (e2e 3146401/03: 3,489 / 3,567) | 3,160 s (3,126 chained) |
| nine15 | 3,520-3,624 s | 3,132 s |
| hardcb cbgraph | 2,995-3,017 s | 3,031 s |
- Margin: nine25 on a 2080 Ti is 0-57 s inside or ~57 s over the limit; the self-imposed 3,060 s margin is missed on 2080 Ti. A channels-last Team B variant (`…_tbgraph_cl`, Team B 36 → 19 ms) projects 3,171-3,266 s on 2080 Ti, 3,114 s on A100 — **unqualified, no signed ZIP, label parity unchecked**; `teamb_overhead.md` not written yet (Sonnet lane, due 10:30).
## Qualification
nine25/nine15: graph vs eager labels identical (0 / 2,655, both orders), official cap250 pair passed (public 0.99592), replay/check OK, NI-V no veto, 16 confirm orders replicated. hardcb cbgraph: pair passed (= cap250 public). No RTX PRO 4500 run exists for any of them.

## Recommendation (to be confirmed at 12:37)
Upload **nine25** (largest, replicated gain; ~+0.9 held-new in four independent order sets, returning +0.55 to +1.2). Prefer nine15 only if the user weighs the 2080 Ti tail: it is ~25 s lighter, 0.2-0.3 pp weaker, a gap with a lower bound above 0 held-new. **Fallback: hardcb cbgraph**, ~600 s inside the limit on every GPU and ≈ +0.2 over cap250. The risk is timeout on a slower host (all nine25 numbers are within ~60 s of 3,600 s on 2080 Ti); a timeout or Failed attempt gives no score and a Failed one costs no quota, so the downside is time, not the best score (cap250 0.8299 stays).

## Could change by 12:37
(1) `teamb_overhead` channels-last/fused result: if a signed ZIP passes label parity and RTX timing, it displaces nine25 as the safe variant. (2) Leonardo dose block 2/pooled `dose.json` and growth screen/freeze (`grow/`; frozen choice only if timing-eligible; +8 ms cap): could add small positive stacking, unlikely to be ready for qualification. (3) Galvani NI-V C/D/E and chained e2e timing (A100/2080 Ti): pending 2FA; a Scherer negative below −0.3 B-new full would veto. (4) Fable round 3 Bet 3 anchor (65011780/81): not a package; no change expected.
