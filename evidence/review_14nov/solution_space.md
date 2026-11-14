# Solution-space review, 14 Nov 2026 (unanchored reader, read-only)

Read for this: `OFFICIAL_CONTRACT.md`; the submodule at the pinned `a66bc23` and the newer `bddaff3` in its object
store (README, `tracks/bci_decoding/`, `benchmark_utils/streaming.py`, `codabench/pages/*`); the Discord reply as
quoted in `docs/SOLUTION_SPACE_MAP.md` §7; `docs/STATE.md`, `docs/LEDGER.md` (index, then §140–§207 sampled),
`docs/SOLUTION_SPACE_MAP.md`, `docs/FRESH_EYES_13NOV.md`, `docs/REVIEW_12NOV.md`, `docs/V6_DESIGN.md`,
`evidence/map_13nov/map_experiments_11_13nov.md`, 616 commits. Nothing was run. Sizes are dev points unless marked
"sealed". Literature statements are from memory; the ones I am unsure of are marked.

## Verdict in five lines

1. I found no untried method family with a credible expected value above +1 sealed. Three items have a tail above
   +1 (each about 10 % or less). The breadth of ideas is not the problem.
2. The search is narrow in three other places: one instrument (ten people, recording-order constants, bars inside
   its noise), one entry route (a posterior added to a fixed-weight mean whose scale the stages depend on), and one
   reading of "shuffled" (three uniform schemes, no mechanism for the loss).
3. The team's own measurement of 14 Nov (§207) says members are nearly irrelevant: +5 accuracy points on a member
   moves the chain by 0.03 to 0.9. A perfect extra view moves it by +6 to +12 (§188). So work on members, cores,
   teachers and weight edits has a low ceiling; work on what the session stage knows has a high one.
4. The largest unexplained number on record is v5.1b's loss under a full shuffle (−3.8 new, §185). No run says
   whether that loss is class-sequence structure (gone for good) or EEG non-stationarity (recoverable). One CPU
   diagnostic decides it (item C below).
5. The documents a new session must read first (`STATE.md` §1, §4, §6, §9; the ledger index; `AGENTS.md` lines 8,
   14, 17, 27; `OFFICIAL_CONTRACT.md` line 36) still describe the recording-order world of 12 Nov.

## 0. The allowed space, from primary sources

| Fact | Source | Note |
|---|---|---|
| One window per call, `(1, 41, 480)`, one final label | `streaming.py`: `current.predict(X[i:i + 1])`, `y_pred.append(pred[0])` | No revision of earlier answers |
| Order is not recording order | Pinned FAQ (`a66bc23`): "There is no shuffling within a stream". Deleted in `bddaff3`, replaced by "the same evaluation order is used for all submissions". Discord 13 Nov: "Please don't assume recording chronology." | **The scope of the shuffle is stated nowhere** (inside a run, inside an interface, whole session). The public `streaming.py` still raises when onsets go backwards inside a recording |
| Adaptation: current session's delivered windows, no labels | FAQ `bddaff3`: "Causal, label-free adaptation using the current and previously delivered windows of the current session is allowed, including statistics-based alignment or gradient updates inside `predict`." Discord: "restricted to the current session's delivered windows, without labels" | |
| Nothing crosses a session | FAQ: "State, windows, statistics, and parameter updates must not cross the session boundary." | See the note on the run-time clock in section 5 |
| Models trained on permitted public data, including recognising people | Discord: "Any models trained on permitted public data are allowed, whatever its training strategy on those public data. So Yes." | Only a quoted Discord answer; not checkable by me |
| Public data may be preprocessed differently; rest runs allowed | FAQ: "You may preprocess the public training data differently"; "Eyes-open and eyes-closed runs may be used for training or analysis" | |
| One `submission.py` plus data files | Guide: "Additional Python modules are not supported." | |
| 60 minutes, one A100 | Terms rule 5 | |
| Best score counts; a failed run costs no slot | Terms `bddaff3`: "Only the best eligible sealed score ... is retained"; "A submission with final status `Failed` does not count." | |
| Cohort | Pinned FAQ: ten represented plus ten absent participants. `bddaff3`: "Exact cohort size, identities, session allocation, and evaluation order remain private." | The 57 % / 43 % split is an assumption |

## 1. Constraints the team works under that the rules do not impose

| # | Constraint | Their sentence | Why it is not a rule |
|---|---|---|---|
| 1 | Everything is an addition to one chain | `AGENTS.md` l.8: "score members and stages in the chain (`replay/`), never alone." `STATE.md` §4: "To test a member: `A.EXTRA[name] = {proto: P}` then `A.system(base, fams, extra=[(name, w)])`" | Guide: `load_model` returns any object. A whole alternative system cannot be scored "in the chain" at all |
| 2 | New information enters as a posterior in a fixed-weight arithmetic mean, and the stages depend on that mean's scale | `STATE.md` §3: "mixed with fixed weights (arithmetic mean of posteriors)". §207: "the stages use the members' scale" | No rule. It is why a better-calibrated or better-fitted combiner loses in the chain (§31, §207 stage 2) |
| 3 | The session stage sees two views only (gaze/event columns; 128 population PCs of core1) | `STATE.md` §3, stages 2 and 3 | No rule |
| 4 | The instrument is ten people leave-one-out plus two returning people, J = 0.5 / 0.5 | `STATE.md` §4: "Dev metric J = 0.5 × new + 0.5 × seen" | FAQ: the public split "remains available for local validation". Terms rule 4 permits other public data, so external sessions may serve as a second instrument |
| 5 | "Preserve the official split and preprocessing" | `AGENTS.md` l.27 | FAQ: "You may preprocess the public training data differently". The team does not use the official 60/20/20 split itself |
| 6 | Returning-person levers are judged by P10 | `STATE.md` §4: "Read returning-person levers by P10's response." | Their own rule (`AGENTS.md` l.17–18) forbids closing on P10 alone; `STATE.md` §5 says the other returning people "are not like P10" |
| 7 | A 45-minute worst case | `V6_DESIGN.md`: "Against a 45-min worst case (every session unrouted), only gC5 with one net per family fits" | Terms rule 5: 60 minutes. This cap chose gC5 over cum10 (§186: "+0.16 mean, +0.45 on the weakest order ... does not fit the run time as is") |
| 8 | Recognised sessions bit-identical to v5.1b | `V6_DESIGN.md`: "Recognised sessions stay exactly as in v5.1b." | Best-of scoring: a worse upload costs only the slot. v6.2 already relaxes this |
| 9 | Bars at +0.3 on an instrument that resolves ±0.3 to ±1 | `AGENTS.md` l.17: "the dev measure's noise (about ±0.3)"; `REVIEW_12NOV.md` §2: "The ruler resolves about one point." | Self-set |
| 10 | The sealed mix is 57 % new, 43 % returning | §186: "The sealed mix is about 57 % new and 43 % returning sessions." | `bddaff3` FAQ: cohort size and session allocation "remain private" |
| 11 | "Shuffled" means one of three uniform schemes (within run, runs, full) | §164 | The organisers named no scheme. A shuffle inside each interface (the metric's cell) is as plausible and is not scored |
| 12 | External data is a teacher or extra training rows | §189: "refuted the external corpora as a teacher, not as extra training people" | Discord answer 3 allows any training strategy. External sessions were never used to tune or learn the session stage, or as a dev instrument |
| 13 | Position-keyed constants | `STATE.md` §3: "early boost in windows 30–60", "from window 60", "from window 10" | Positions in a delivered order carry no recording meaning. Not a rule question; a recording-order design carried over |
| 14 | Uploads held back | `STATE.md` §1: "no new package candidate is built and no upload decision is taken" | Two uploads, seven days. The first of them is also the only measurement of the new-people block |

## 2. Closures that do not hold up

"Retest" always means: on v6f or v6.2, package-style, under within / full / runs, new and returning people reported
separately.

| # | Direction (section) | Why the closure is weak | Cheapest decisive retest |
|---|---|---|---|
| a | Interface as a stage variable (§103b: per-interface rebalancing −0.41) | Oracle on the "grid chain without routing", recording order. In recording order an interface is a contiguous run, so the 30-window rebalancer already is per interface. Under full or runs shuffle the interfaces interleave, and the metric is per interface cell | `shufcombo` with the true `context_id`: rebalancer counts and view standardisation per interface; full 0–3, runs 0–3. CPU, stored outputs. If ≥ +0.4, train a two-class interface detector (should be easy: different displays and timing) |
| b | Run type as a stage variable (§109: −0.10 / −2.98) | Only complete separation of statistics was tried, in recording order. Under shuffle calibration and feedback windows interleave from window 0 | Oracle run type as an additive offset in the templates (class mean + domain offset, shared covariance), not a split |
| c | Fitted log-linear pool (§31: +0.3 to +1.2 at mix level, −1 to −2 in the chain) | Exploratory, v3 chain, recording order, stage constants not re-chosen. §207 shows the mechanism is scale, not content | Fitted pool, then a monotone map that matches its margin distribution to the arithmetic mix's (fitted on training people), stages unchanged |
| d | "Member weights and stage constants sit at an optimum" (§31, §66, §69) | v3, recording order, B2_pool. §166 re-chose stage constants only, for v5.1b. Nothing re-chose member weights with the v6 additions under shuffle. §207: six of eleven slots have leverage ≤ 0.02 | Leave-one-slot-out table for v6f under the three schemes (11 runs of 2.5 min), then nested re-weighting |
| e | PLS-DA view for TSEM (§74: +0.45 to +0.60 against a +0.25 null) | Two exploration bases, never taken further, recording order | Same view inside v6f, eight orders |
| f | Perfect recognition from window 0 is worse than the real gate (§46a: −0.24) | Two people, recording order. A perfect oracle that loses says the routed path is wrong in its first windows, not that recognition is saturated | Oracle identity from window 0 on v6.2 under shuffle; if still negative, per-window loss on delivered positions 0–10 by component |
| g | Returning-person dose (§19 +0.19, §22, §95, §111 β 0.15) | Each failed a clause on P10 alone (§19: "0.26 = one window"). §19's own reading: "A per-person trust weight is the obvious refinement" and it was never run | Section 3, item K |
| h | Order-free stages (§181: "no ceiling under within", "undecided" under full) and time weighting (§170) | The remaining loss (−1.6 of −3.1 under full) has no mechanism. "Temporal structure that recording order offers" is an inference, not a measurement | Section 3, item C |
| i | Learned stage (§70 confounded, §98 below chance on one seed and "not debugged", §141 −8 to −12) | All trained on ten people's sessions or on synthetic ones | Section 3, item D |
| j | Cue-order prior and cue generator (§9, §55, §27) | Not weak: moot. The organisers shuffled to remove exactly this. Mark "moot under shuffled delivery", so nobody re-opens it | none |
| k | Order fingerprint; order-robust re-tune (index: "every reordering hypothesis rejected", §2, §3) | Wrong as written: §62 is withdrawn and §3 never ran | Replace the index row (section 5) |

Closures I checked and would leave closed: session clustering from scratch (§30); similarity routing and person-type
experts (§16, §67, §110, §116: four forms); virtual people at core level (§46b–§49: §146 shows core gains carry to
the chain at 0.95); Scherer as training rows (§7, §21, §53, §191: retested on today's chain); frozen foundation
embeddings (§71: five models, equal to a random network); MixStyle, DRO, focal (§153, §93, §87); mirrored MI (§5 A25).

## 3. Families: what a shuffled, unlabelled, one-window stream permits

Covered and not worth more (said plainly, since the brief lists them): test-time adaptation (statistics, EM
templates, pseudo-label and consistency fine-tuning are all in; entropy and label spreading failed, §91, §89);
Riemannian and CSP systems (core1 is one; alone it gains +0.8 from the stages, §48); foundation models; domain
generalisation objectives; participant synthesis; mixture of experts; per-session class prior (the rebalancer).
What BCI Competition IV 2a (filter-bank CSP) and BEETL 2021 (per-subject latent alignment, ensembles; the labelled
few-shot part is not allowed here) rewarded is present. I am unsure what won the 2025 EEG Foundation Challenge.

| # | Family | Why it could matter here | Smallest sizing experiment | What told them not to |
|---|---|---|---|---|
| A | **A second instrument: external three-class sessions** for stage, fine-tuning and order decisions | Every choice of the last 48 h is inside the noise (v6.2 against v6f + ERF: 0.03; s15: +0.14; label source: ±0.2). Nested selection on ten people loses 1.4 (§66). Stage and fine-tuning mechanics are task-generic | A reduced chain (tangent-space core, two SessionBN nets, rebalancing, TSEM, gC5) on three-class subsets of BNCI2014-001, PhysionetMI, Scherer and one many-session corpus. Validity check first: it must reproduce the sign of five known PROTEUS results (stages +, fine-tuning + under shuffle, RETUNE 0, Winf −, MixStyle −) | `AGENTS.md` l.8 and l.27; index row "external corpora: nothing usable" (§33); three negative Scherer poolings (as rows, a different use) |
| B | **Domain-conditional session model** under shuffle (interface × run type as an additive latent in the templates) | Under full or runs shuffle every prefix mixes four domains. The metric is per interface. Returning people lose 8 points in Graz feedback runs against calibration runs (§105) | Oracle forms a and b of section 2 | §103b, §109, §168 (hard splits, recording order) |
| C | **Class-sequence-preserving shuffle** (a diagnostic, not a method): permute windows only among windows of the same class, session-wide | Separates the two explanations of the −3.8. If v5.1b loses about 0, the loss was class-sequence structure, no stage can recover it, and order-robust stage work stops. If it loses about 3.8, it is EEG non-stationarity and B has a real ceiling | One `shuffle.py` scheme, six bases, CPU. Add the per-window loss table (delivered-position decile × run type × interface) from stored outputs | §181's reading was accepted without it |
| D | **A learned set stage, meta-trained on external real sessions** (permutation-invariant net over the delivered windows' member posteriors and a few features) | The stage holds the leverage (§188, §207). It is hand-built and tuned on ten people. An exchangeable stream is the set-function case. External corpora give hundreds of three-class sessions for the generic operation "refine noisy posteriors over a balanced set" | Train on leave-one-person-out posteriors of a simple model on PhysionetMI three-class sessions; beat rebalancing + TSEM on held-out external people first. One GPU hour decides whether to go on | `FRESH_EYES` 2.7: "I would not spend more here"; §98; `REVIEW_12NOV` "ten people are too few" (true of PROTEUS, not of the external corpora) |
| K | **Per-person offline dose for returning people** (stage strength set per public person from public labels) | §15: gain follows the person's weakness (Spearman +0.52 to +0.58). §79: P04 wants β 0.15–0.40, P10 0–0.05. Label-free session trust failed (§79, §80), but each public person's weakness as a new person is known offline and is a legal constant | β_j = a · (leave-one-out deficit of person j), a fitted on the ten-person within-session proxy (§41, §82), read on P04 and P10 under shuffle | `STATE.md` "Read returning-person levers by P10's response"; §95 "β 0.10 stays"; §41 "never beats one recipe" (that was the choice of space, not the dose) |
| M | **Run time moved from frozen members to fine-tuning** | §207: most frozen slots have leverage ≤ 0.04, yet fine-tuning the same slots gives +0.45 to +0.58: the gain is session information, not member accuracy. 45 of 60 minutes is the self-set cap | Ablation table (2d), then cum10 or two nets per family inside 55 minutes | "Extra members lower the chain" was read as "the set is optimal" |
| N | **Both remaining uploads as a hedge across the scheme** | v6.2 leads under within, v6f + ERF under full; best-of scoring keeps the better one | none | `FRESH_EYES` New 4 said this; nothing selects candidates that way yet |

## 4. Ranked top ten (sections 2 and 3 together)

Honest uncertainty: every dev size below carries ±0.5; dev to sealed has been 0.4 to 1.0.

**Tail above +1 sealed (each about 10 % or less to exceed +1; expected value in brackets)**

1. B, domain-conditional stage, with oracle forms 2a and 2b first (expected +0.2 sealed, up to +0.8 if the stream
   is full or runs; about 0 if it is within).
2. K, per-person offline dose for returning people (expected +0.15 to +0.3; more if several returning people are
   weak as new people).
3. D, learned set stage on external sessions (expected +0.1; tail +1 to +2; three days; the explore third only).

**Small but cheap**

4. C, class-sequence-preserving shuffle and the per-window loss table (0 direct; decides 1 and all order work).
5. 2d, slot ablation and re-weighting of v6f under shuffle; M, the freed run time into cum10 (+0.1 to +0.3 dev).
6. 2e, PLS-DA view in v6f (0 to +0.3 dev).
7. 2c, fitted pool with scale matching (0 to +0.3 dev).
8. 2f, oracle identity from window 0 on v6.2 (0 to +0.2 on returning people).
9. A, second instrument (indirect: makes 5 to 7 decidable; one to two days).
10. N, the two uploads as a scheme hedge (option value; costs nothing but the choice).

Free, outside the ranking: ask the organisers one question, "is the delivered order a permutation of the whole
session, of each interface, or of each run?"

Checked by search, 14 Nov: the ledger from §164 onwards contains no mention of the interface, no class-preserving
shuffle, no loss table by delivered position, and no external corpus used as a dev instrument.

## 5. Statements that are stale, wrong or narrowing

### One compliance point found on the way (not a search question)

`lean/src62g/finetune.py` l.13–14 and l.130–134 (same in `src62h`): `FT_CLOCK = {"t0": None, "stopped": False}` is
module-level, with the comment "lives here, not on the model (the evaluator deep-copies the model per session)".
Once 2,700 s have passed it switches fine-tuning off for every later session. That is state crossing the session
boundary and changing later predictions (FAQ: "State, windows, statistics, and parameter updates must not cross
the session boundary"; `FRESH_EYES` 1a: "Module-level state ... prohibited"). It also makes the score depend on the
worker's speed, which the ±2σ audit rerun would expose. It is inert below the threshold. Replace it with a
per-session limit that needs no memory of earlier sessions (the per-session refit cap already exists), or accept
the timeout risk knowingly.

### `docs/STATE.md`

| Line | Statement | Problem | Replacement |
|---|---|---|---|
| 1 | "as of 13 Nov 2026, 00:20 UTC" | The file holds later content and is two days of work behind | Date it; state the current candidates (v6.2, v6f + ERF, v6.2gL) |
| 14–16 | rank 8th, board of 12 Nov | Stale (about 10th; third 0.831) | Current board, or drop the ranks |
| 21–22, 271 | "Research window ... until about 14 Nov 02:00 UTC" | Over | Delete |
| 23–25 | "Galvani is unreachable until the user logs in" | Galvani ran jobs on 13 Nov | Current access state |
| 26–37 | "What is measured against v5.1b as uploaded (13 Nov, the only reference from now on)" | Every number in the list is recording order (§151, §152, §155) | The morning table per scheme (§203–§206) |
| 34–35 | "The ceiling of the eye-net kind of view is about +3.7 more (oracle...); better views are the explore lever" | The oracle is "0.6·one-hot(true label)" (§188): the ceiling of any perfect view, not of eye signals. Eye-view accuracy has leverage 0.006 per point (§207); its limit is error correlation | "A perfect extra view is worth +6 to +12; the eye net's accuracy is not the lever, its within-run error correlation is (§188, §196, §207)" |
| 47 | "Inside a session everything causal is allowed" | Wider than the text (`FRESH_EYES` 2.2 said so; still there) | "Label-free adaptation on the session's delivered windows; nothing fitted on shipped labelled data during a session" |
| 89 | "Rebalancing ... worth 5.75 J" | Recording order; §181: the 30-window form is "consistent with it exploiting temporal class blocks" | Give its value per scheme |
| 93–96 | boosts and gates "in windows 30–60", "from window 60" | Positions in a delivered order | Say "delivered windows" and that no recording meaning attaches |
| 115–131 | J, six bases, `replay/check.py` reference values | Describes the v3 chain in recording order as the instrument | Primary protocol: package-style, three schemes, new and returning separately; `check.py` guards the v3 replay only |
| 143 | "Treat anything below +0.3 as noise." | The same file (l.176) and `REVIEW_12NOV` say ±0.3 to ±1 | "Below +0.5 nothing is decided on ten people; +0.3 is the floor for a null arm, not the resolution" |
| 145–147 | "Read returning-person levers by P10's response." | Contradicts l.178–179 and `AGENTS.md` l.17–18; it closed §19, §78, §111 | "Report P04 and P10 separately; neither decides alone" |
| 158–160 | "core1 rises +8 points per doubling" under large ceilings | Chain level is +4.7 (§146, §189), and no data exist to buy it (§189a) | "+4.7 per doubling at chain level; no public corpus supplies such people" |
| 161–163 | "an independent member at 70 % is estimated at +3.1" | Synthetic. Measured leverage of the teacher is 0.08 per point (§207) | Add the measured leverage |
| 166–167 | "aligning a session with its true class means lifts a simple classifier from 64.6 % to 89.1 %" | Fitted on the test windows' own labels; the causal true-label form is 68.7 (§104) | Delete from "large ceilings" |
| 170–171 | "Choosing per session ...: +1 to +2 with an oracle" | §193: "any oracle over three arms gains from noise alone"; its ceiling was +0.41 | Add that most of it is selection noise |
| 222–224, 266 | "v5.2 / v5.2b are not candidates" and "Next upload (v5.2 or a later candidate) ... v5.2 is built and being validated" | Contradict each other; both stale | One line naming the v6.2 family |

### `AGENTS.md`

| Line | Statement | Problem | Replacement |
|---|---|---|---|
| 8 | "score members and stages in the chain (`replay/`), never alone" | Forces every idea to be an add-on; `replay/` is the v3 chain in recording order | "Score in the full current system under shuffled delivery; score a whole alternative system whole, with its own stages" |
| 14 | "`python replay/check.py` must print OK before any chain result is trusted." | It checks v3 in recording order and says nothing about v6 under shuffle | Name the check that guards the current package path |
| 17 | "the dev measure's noise (about ±0.3)" | ±0.3 to ±1 by the team's own reviews | "about ±0.5 on ten people" |
| 27 | "Preserve the official participant split, preprocessing and metric" | The split is not used; different preprocessing of public data is explicitly allowed | "Preserve the official metric and the delivered tensor; splits and training-side preprocessing are free" |
| 38 | "Prefer Galvani for anything heavy." | `STATE.md` says Leonardo | One current line |

### `docs/LEDGER.md`, index

| Line | Statement | Problem | Replacement |
|---|---|---|---|
| 18 | "Anything below +0.3 is inside the noise" | As above | ±0.5 |
| 28–157 | The index ends on 12 Nov | No index for §66–§207; a reader of "the index" never meets the shuffled protocol | Header: "Every row was measured in recording order on the v3 or v4.1 chain; the stream is shuffled (13 Nov). Stage rows are untested under the current protocol." Add an index of 13–14 Nov |
| 71–72 | Cue-order prior; cue generator: "Closed" | Moot, and the organisers shuffled to remove them | "Moot under shuffled delivery; do not re-open" |
| 73 | "Order fingerprint of the sealed stream; order-robust re-tune: every reordering hypothesis rejected (§2, §3)" | §62 is withdrawn (§164); §3 never ran | "Withdrawn: the stream is not in recording order. See §164–§206" |
| 80 | "State across sessions, slow potentials, feedback in online runs, external corpora: nothing usable" | The teacher (+1.4 J) and the gate (224 external people) are external corpora | Remove "external corpora" |
| 81 | "runtime is not binding" | It is now: it chose gC5 over cum10 | "binding since fine-tuning (§120, §186)" |
| 98 | cand_v51_own "Built and ready (not uploaded) ... Sealed estimate 0.820–0.825" | Uploaded 12 Nov, 0.8198 | Move to "Adopted" with the sealed result |
| 105–106 | Eye branch and fine-tuning "not built" | Both are in v6 | Update |
| 110 | "Open: None on 12 Nov 06:05 UTC" | Stale | Delete |

### Other documents a new reader meets

| File, line | Statement | Problem |
|---|---|---|
| `OFFICIAL_CONTRACT.md` l.36–37 | "Runs follow BIDS run number; windows follow recording time." | True of `a66bc23` only. `AGENTS.md` l.5 sends every new session here. Add the 13 Nov answer |
| `docs/V6_DESIGN.md` l.78–82 | "the *within* shuffle is the primary dev column" | `STATE.md` l.69 and §205: "No scheme is primary" |
| `docs/LEDGER.md` §186, rank 1 | "Eye-view quality ... ≤ +2.2 sealed ... still the largest measured ceiling" | Same wrong oracle as `STATE.md` l.34 |
