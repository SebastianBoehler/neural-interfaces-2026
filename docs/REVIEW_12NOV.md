# Critical review, 12 Nov: what moved the score, what stopped, and what holds us back

Sources: the 197 commits and every document under `docs/` (read-only pass by a second reader), the replay, and two
diagnostics run for this review (ledger §66 WEIGHTS, §67 PTYPE). Standing: 0.8162 sealed, rank 10; leader 0.847.

## 1. What produced the gains

| When | Step | Size | How it was found |
|---|---|---|---|
| 18 Oct | Causal recentring, slow band, net ensemble, rebalancing | +7, +7, +7, +2 (dev LOSO) | Reading what the rules allow; a broad feature probe; no gates |
| 30 Oct | SessionEM | +2.3 | A ground-up review asking for an independent low-dimensional view |
| 1–2 Nov | External teacher | +1.4 J | An oracle: labels with independent errors pay, the chain's own do not |
| 5–6 Nov | SessionBN nets, core1, Hilbert core, TSEM | sealed 0.7795 → 0.8197 | **Upload 1 came back 3–4 points under expectation**; it set off an approach tournament and a redesign of the core in 36 hours |
| 7 Nov | Routing for returning people | sealed +0.43 | Follow-up of the same review |
| 11 Nov | Identity gate | sealed +0.27 | |
| 12 Nov | Own-space stage (v5.1) | dev +0.89, all of it P04 | Not uploaded |

Every step of a point or more came from one of three sources: an outside measurement that contradicted us, an oracle
that located missing information, or the redesign of a whole component. Three of the winners (session normalisation,
re-referencing, routing) had been closed by argument a week before they shipped.

## 2. What stopped on 7 Nov

Since 7 Nov about 150 pre-registered runs produced two adoptions, both for returning people and both P04's. The runs
changed in kind: almost all are one change to a frozen chain, scored on cached member outputs within minutes. That
screen has three built-in biases against anything new.

1. **The baseline is tuned on the people it is scored on.** Choosing the chain's constants on half the people and
   scoring the other half loses 1.4 points (§66). A new component enters at an untuned weight against constants that
   carry that in-sample advantage.
2. **The ruler resolves about one point.** Ten people, per-person changes with a standard deviation of 1.5–3.5, P04
   30 % of J and 3 of about 106 sealed sessions. Gates of +0.3 to +0.5 sit inside that noise.
3. **New members were judged with the old stages.** A new core entered as posteriors only; the session stage and the
   routing kept looking through the old core's features.

The chain is not mis-tuned (§66: the shipped constants are at the 98th percentile under a sealed-like weighting), and
nothing found on 12 Nov says a knob is left.

## 3. Where the missing points are

- **Inside our design, perfect information is worth at most 3 points.** True labels for in-session learning: +2 to
  +2.9. The best constants per person: +2.4, most of it selection noise. The chain already matches a supervised
  model trained inside the session on our features. More adaptation of the same representation cannot close a
  3-point gap; this is why 60 session-stage runs returned nothing.
- **People differ, and the loss is concentrated (§67).** As a new person P04 scores 53.6 %; the other nine average
  83.1 %. P04 is not noisy: their own earlier sessions predict them at 76 %, nobody else's model above 46 %. Four
  people (P01, P02, P07, P08) transfer to each other at 56–69 %. How well others transfer to a person predicts the
  chain's score on them (Spearman +0.78). One person in ten of that kind costs the new-person mean 3 points, and we
  have no label-free access to their signal.
- **The start of a session.** New people are at 72.7 % in windows 0–15 and 84–86 % late in the session (without P04).
- **Returning people are unmeasured.** The dev measure has two of them, one the outlier. Eight returning people have
  one public session; what a personal model is worth for them on the sealed set is unknown (estimate +0.1 to +1.2).
- **Members trained on the same ten people share person-systematic errors** and do worse in the chain than random
  noise of equal accuracy. A member with independent errors at 70 % would be worth about +3; the only one we have
  (the teacher) is 50 % alone.

## 4. Limits we put on ourselves

| Limit | What it cost |
|---|---|
| Uploads rationed: 4 of 7 used, three built probes never sent | The sealed split between new and returning people and the teacher's sealed value are unknown. The one upload that surprised us produced the largest gain of the project |
| One chain, extended by additions | No second system was ever built and compared whole |
| The "57 % new, 43 % returning" cohort comes from a text the organisers withdrew | All positive work since 7 Nov targets a block whose size we assume |
| The warm-up board was dismissed in October | The weak core surfaced a week late |
| Runtime: 19 of 60 minutes used | The leader's jump came with a fourfold runtime |
| Gate habit and "do not re-propose" lists | Four listed items later shipped |

## 5. What follows

1. **Measure before optimising further.** v5.1 on the sealed set is the only available measurement of the returning
   block. It does not fix a baseline: the board keeps the best score, and the stage's loss is bounded. Reading rule
   as in §63. This is the user's decision.
2. **Stop single-knob screening on the frozen chain.** A new run needs one of the three sources of §1.
3. **Two system-level builds, each scored as a whole system with its stages re-fitted, on people halves:**
   a new core with the session stage and routing looking through its features; and frozen foundation-model
   embeddings as a view for the session stage (the only route left to information from other people; so far every
   foundation model was fine-tuned on our nine people and gated alone).
4. **P04-like people:** a label-free detector for "nobody transfers to this session" exists (agreement, +0.83). What
   to do once detected is open; it is the largest located loss.

## 6. What followed the review (12 Nov afternoon; ledger §68–§78)

Every limit of §4 was answered with a build, scored on all six bases with person intervals:

| Limit | Build | Result |
|---|---|---|
| One chain | two complete chains fused late (§68); a learned sequence stage (§70) | −0.7 to −6; −10 to −15 |
| Baseline tuned on its test people; fixed-weight screening | all 64 stored candidates re-screened with the chain co-tuned, cross-fitted (§69) | best +0.15; none was closed wrongly |
| Information only from our ten people | frozen foundation embeddings as a session-stage view (§71) | equal to a random network of the same shape |
| New members judged through the old stages | the nets' own features as a session-stage view (§72–§73) | +0.5 on the choosing nets, +0.20 on fresh nets, +0.07 over six bases: refuted |
| Stage strength set on 6 Nov | TSEM λ re-chosen (§75); per-session choice of variant (§76) | cross-fit −0.07; oracle +1.0 to +1.4, no label-free criterion |
| P04-like people | per-band transfer map (§77) | no band or family carries P04 from others (best 49 %) |
| Returning people | personal nets on top of the stage (§78) | +0.2 seen, P04 only |

What this establishes: on dev, the system is at the limit of what our ten people can teach it. A random second view
moves new people by ±0.4, the exploration bases overstate stage effects by about twice, and the head-room that
exists is per person (P04 76 % from their own data, 49 % from anyone else). The one block dev cannot measure is the
sealed returning people with one public session; that is measured only by an upload of v5.1.


## 7. Second review, 12 Nov evening: two independent reviewers (literature; internal audit)

**Where they agree with each other and with §1–§6.**
- No published label-free method for cross-subject or cross-session transfer is missing from the chain: recentring,
  per-session latent alignment (SessionBN, the BEETL winner's idea), Bayesian/EM test-time adaptation (TSEM),
  online test-time training (§60, §81), domain generalisation (§93), foundation models (§71). Published gains of
  +2 to +6 are measured on single models; on a rebalanced ensemble with EM stages they do not stack.
- There is no +3 left for new people on dev. §83/§88/§89 bound what label-free methods can do; §88's 78 % is a loose,
  non-causal upper bound (the own model also saw later runs).
- The large unmeasured block is returning people. If the sealed set holds the later sessions of the ten public people
  (about 43 % of cells; the official FAQ keeps the allocation private, so this is plausible, not given), a strong
  personal system worth +6 there is +2.6 on the board, which would explain most of the leaders' margin.

**Corrections to our framing.**
- **The leader is one team.** The public leaderboard page lists ranks 1 and 2 both as team ETRI_EnSeption (two
  models, 0.847 and 0.846) and rank 3 as sungjunwang (0.830). The page lists submissions, not teams: counted by team
  we are 9th, and third place by team needs about 0.826. STATE §9's "two accounts" item is withdrawn.
- **Dev "new" is mostly first sessions of untrained users**, while most sealed new-person sessions are later
  sessions of new people (PROTEUS trains users with feedback). The hard early third (§89) may be smaller there.
- **Uploads were rationed and never used as probes.** No upload ever isolated the teacher, the routing or the
  returning block; v5.1 has been ready since the morning of 12 Nov.

**Axes never tested, with plausible head-room.**
1. Re-referencing for core1 (surface Laplacian; CAR + Laplacian): 0 to +1 for new people. Closed by argument on
   30 Oct. Running now (§99).
2. Extra window offsets as training rows for the *personal* models (start0.5 never used anywhere; offsets were only
   tried for the population core, where "only P04 gained"): small, returning people only.
3. Gate recall against the gap between public and sealed session (sealed sessions are 1–5 sessions later).
4. A labelled Procrustes oracle (recentre, stretch, rotate with true class means) to close the alignment family on
   evidence rather than argument.
5. Order robustness: risk control, not upside (the organisers dropped the chronological-order guarantee).

**Mismatches the audit found.**
- `replay/check.py` checks the v3 chain only; v4.1 and v5.1 have no reference check that must print OK.
- The own-space stage swallows every exception (`submissions/ensemble_v5/submission.py` 850–861) and skips flat or
  missing channels: a montage difference on the sealed run would silently turn v5.1 into v4.1, and a "+0.0" result
  would then wrongly stop the returning branch under §63. Before any upload of v5.1 a variant that reports, at
  session end on stderr, how many windows were staged is built (a new folder; the pinned file stays unchanged).

**Plan for the next 24–48 hours (no upload before it).**
1. Tonight: §99 Laplacian core (about 1.5 h, local); the Leonardo lines (§91, §94, §96, §98; by about 21:00 UTC).
2. Then, in order: personal models with offset crops (returning); dev split by session index (framing check); the
   labelled Procrustes oracle (closure); a v5.1 replay check in `replay/check.py`; the loud-failure v5.1 variant.
3. Decision on 13 Nov evening: upload the best checked candidate — v5.1 with loud failures, plus anything that passed
   on all six bases — as the measurement of the returning block; a v5.2 with personal nets follows only if it pays.
