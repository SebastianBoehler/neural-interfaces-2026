# Research brief 2 · Fri 20 Nov 2026 evening

Goal: find methods (any field, especially very recent work — NeurIPS 2026 papers are going public ahead of the
December conference, ICLR 2027 submissions, arXiv of the last weeks) that attack OUR measured bottlenecks and have not
been tried here. The leaderboard (1st 0.859, 3rd 0.848; us 0.8299) shows more is possible on the same data, so the
limit is our method, not the data. Constrain only by the official rules (OFFICIAL_CONTRACT.md; 3,600 s on one RTX PRO
4500, one label per window from current + earlier delivered windows of the session, nothing across sessions, any
public data and pretrained model allowed, training is free).

## Measured bottlenecks (evidence in evidence/wave_20nov/ and evidence/audit_20nov/)

1. **New people (~57 % of sealed, ~75 % of error)**: five people near 69 %; P04 ~50 % without labels, 85–90 % with one
   labelled session → the signal holds each person's own class map, but label-free in-session methods recover 0–3 pp.
2. **The mixer absorbs member gains** (lane Z): one family slot ≈ 1/16 of the mix; real member improvements are
   redundant (right where others are already right). Gains that are right where the mix is wrong pass and even amplify.
   Mixer re-weighting (R1) gives nothing honest.
3. **Session-level stages carry cap250** (rebalance −2.4 pp and EM/TSEM −2.3 pp if removed); they pass injected gains.
4. **Returning people (~43 %; ~87 % of them seen in ONE public session)**: keeping adaptation on after recognition
   (R4 form) +3.7 pp on a far-from-ceiling cross-day set (NETBCI, 17/19 people), but negative on near-ceiling P04/P10.
5. **What failed**: foundation models (REVE, CBraMod, Pratibha-2, BrainCodes, NeuroRVQ) frozen, fused or fine-tuned;
   extra data (81 MI people, Shin, Scherer); virtual subjects; clean-slate systems (6–15 pp below); gradient TTT on
   strong systems; person-expert weighting; learned in-context adaptation (+2 on a weak base, ≤ prefix BatchNorm).
6. **What works a little**: hard pair, CBraMod member, lite5 TTT, a post-mixer within-window dispersion correction
   (helps new people, hurts recognised returning people).

## Output
Each researcher writes evidence/council_20nov/research2_<you>.md (≤180 lines): ranked ideas with mechanism, why it
should beat what failed (tie to a bottleneck above), rule check, runtime fit, expected size (honest), and a decisive
test runnable on Galvani/Leonardo by Sat 21 Nov ~10:00 CET. Cite primary sources with dates. Commit only your file
(`git pull --rebase --autostash && git push`, no AI author line, never force-push). Never upload. No personal or
project-identifying info in web queries.
