# Map: what the rules allow against what we tested (13 Nov 2026)

**Purpose.** Asked by the user on 13 Nov: set the experimental space (git history, the documents, the wide waves
since 11 Nov) against the allowed solution space, to find approaches we missed, scopes we narrowed too early, and
paths we closed or set aside on weak evidence.

**How it was made.** Three independent readers, each read-only: (1) the official rules, FAQ, config and evaluation
code at first hand, including the newer revision `bddaff3` that sits in the submodule's object store; (2) all commits
and documents up to 10 Nov; (3) all 119 ledger sections from 11 to 13 Nov, each tagged by how it was tested. Their
full reports are in `evidence/map_13nov/`. Nothing was re-run for this map; numbers are quoted from the record.

**Reading aid.** "Chain" means scored inside the full system. "Posterior-only" means a new part entered as its output
probabilities while the session stages and the routing kept using the old core's features.

## 1. The allowed space: what we may do

| Area | The rules say | Do we use it? |
|---|---|---|
| Public PROTEUS, all 14 sessions, all labels | Allowed for training and selection | Yes |
| Rest runs (eyes open / closed) | Explicitly allowed for training or analysis | **No.** REST4 used pre-cue periods only |
| Rest of the recordings (0–1 s cue period, 5–8 s trial end, inter-trial, raw 500 Hz) | Allowed ("you may preprocess the public data differently") | Barely: shifted crops (negative), pre-cue rest as a class (+0.1 to +0.3) |
| BIDS entities for training (person, session, interface, run, calibration or feedback run) | Allowed | Partly: as oracles for stages; person as an adversary |
| External public data, pretrained models | Allowed if their terms permit; must be declared | Yes: 7 MI corpora (teacher); cognitive corpora tried |
| Gradient updates inside `predict` on the session's own windows | Explicitly allowed | Measured (+0.36 for new people), not built |
| Keeping every earlier window and refitting from the full history each call | Allowed | Yes (stages, own-space stage) |
| State across Graz and BrainHero in one session | Allowed | Yes |
| State across sessions | Forbidden | Not used |
| Run time | 60 minutes for the whole run, spent however we like | 21.5 minutes used |
| Package size | Up to about 15 GB | 62 MB used |
| CPU and GPU, threads | Unrestricted | Mostly sequential |
| Failed run | Costs no quota, daily or total | Not used as a safety net (see §5) |
| Leaderboard | Best score kept automatically | — |
| Output | One hard label per window; no revision; a label outside 0–2 or a class absent from the cell is penalised | Respected |
| Audit (top 3) | Organisers rerun **our training pipeline** and rescore; must land within ±2σ | Rebuild recipes exist for v4.1; none yet for v5.x parts |

**Not guaranteed any more (since 10 Nov):** the order of windows inside a sealed session. Our chain loses 3 to 5
points under a re-ordering; the order-robust re-tune (ledger §3) never ran.

**Unspecified, and it matters** (questions for the organisers, §6): using labelled public data at evaluation time;
using protocol regularities (cue order, class balance per run); identity inference (our permission is a Discord
answer); what σ is in the audit; whether the audit reruns external pretraining.

## 2. Coverage by axis

Legend: **dense** = many runs, scored in the chain with controls. **thin** = few runs, one configuration, or scored
only alone or posterior-only. **none** = no run.

| Axis | Coverage | What the record shows |
|---|---|---|
| Window features for the core | dense, but posterior-only since 6 Nov | The 6 Nov redesign gave about +10 alone after 19 days without a redesign. Every later core variant entered as posteriors only |
| Net architectures | dense as extra members | About 30 families; judged as one more member at a fixed weight |
| Foundation models | thin to dense | Five models, all weak alone (35–54 %) |
| Members from other populations (teacher) | **thin** | The only member with independent errors. Design fixed 1–2 Nov, judged then on a measure that was 65 % one person (P10). Never redesigned for the current chain, never isolated on the sealed set |
| Training data: more labelled people | dense, mostly alone | +8 points per doubling of people at core level; nothing substitutes; chain-level slope unknown |
| Training data: unlabelled PROTEUS (rest runs, inter-trial, raw rate) | **none** | Proposed in five reviews since 20 Oct; never run |
| Session adaptation from statistics and templates | dense | At an optimum for this chain (about 60 runs) |
| Session adaptation by gradient steps | medium | Label-limited: true labels +1.6 to +2.9, own labels +0.36 at best |
| Alignment of a session to the population | thin | One form (Procrustes) as an add-on member; the pre-registered follow-up inside the template stage did not run |
| Protocol structure (run balance, cue order, run type, interface) | dense, as oracles | Rebalancing carries 5.75 J; everything beyond it is ≤ +0.4 even with true metadata |
| Returning people | dense on two people | The only axis that moved the sealed score since 7 Nov (+0.27, +0.36). Dev has two returning people, one an outlier |
| New people as individuals (types, similar people) | medium | Selection among sources always loses to pooling |
| Weak sessions (P04-like) as a case of their own | **none** | Session confidence tracks session accuracy (r about 0.8), and fine-tuning gains sit in weak sessions. No policy keyed on it was tested |
| Combining members | dense | Fixed weights at an optimum; per-session choice has +1 to +2 of oracle headroom and no label-free rule finds it |
| **Stacking small positives into one system** | **none** | One test of two returning-person levers (negative). About ten replicated small gains were never scored together |
| Whole alternative systems | thin | Six tried, each in one configuration, each scored as an add-on to the frozen chain; one failed below chance on one seed and was not debugged |
| Dev measure itself | dense on diagnosis | Resolves about ±0.3 to ±1; bars of +0.3 to +0.5 sit inside it |
| Sealed uploads as measurements | **thin** | Five uploads, none isolating a component; new-versus-returning split of our sealed score unknown |
| Run time and package size | thin | 36 % of the time and 0.4 % of the size used |
| Robustness to window order | **none** | Risk named on 10 Nov; no run |

## 3. Lock-ins and how they aged

**Five of the largest gains came from directions that had been closed earlier on weak evidence:**

| Direction | How it was closed | What happened |
|---|---|---|
| Session normalisation inside the nets | Extended by argument from a narrower test (30 Oct) | Shipped 6 Nov |
| Re-referencing (CAR) | By argument, no run | In both shipped cores |
| Self-labelled session templates | Tested on a 56–65 % pipeline | TSEM, +1.7 J on 6 Nov |
| Recognising returning people | Four tries, always spectral identity features | Worked with class-direction features (7 Nov), then with the identity gate |
| Core feature design | Tuned by a point at a time on 18–19 Oct, then left | Redesign on 6 Nov |

**Lock-ins still in place and never re-opened under the current chain:**
1. **Teacher design** (frozen external nets → head outputs → small logistic map → one member at weight 5). Stronger
   teachers were rejected on the old measure: more MI corpora lifted the teacher alone by 5 points "with no ensemble
   gain" on a score dominated by P10. Cognitive-task heads as a separate member were closed without a run.
2. **New parts enter as posteriors only.** The 12 Nov review asked for "a new core with the session stage and routing
   looking through its features". That build never ran.
3. **One window, one member mix, one chain.** Alternatives were scored as additions to it.
4. **Gate habit.** At least nine directions were dropped within 0.01 to 0.1 of a pre-set bar that itself sits inside
   the noise of the measure (examples: +0.29 against +0.30; +0.36 against +0.40, replicated on every base).
5. **Decisions resting on P10 alone.** Several returning-person levers were dropped because P10 moved by −0.1 to −0.6.
   P10 is at 91 %. Upload 5 suggests the other returning people are not P10-like (about +0.6 per session on the
   assumed cohort).
6. **Uploads rationed, never used as probes.**

**Closed on an early pipeline and never retested on the current one:** Sinkhorn balancing (showed a later-session
gain on 20 Oct that was set aside as noise); xDAWN / ERP-covariance member; MixStyle on SessionBN nets (the October
run was a silent no-op for one family); EEGNeX in the trainable-filter slot (J +2.17 against +1.65 on 5 Nov, never
confirmed); per-session alpha-peak band shift (never run).

## 4. The gaps, ranked

Ranking is my judgement: size of the possible gain, strength of the evidence that it exists, and how badly it was
tested. Sizes are from the record; none is a promise.

| # | Gap | Why it is on the list | Cost |
|---|---|---|---|
| 1 | **Score the replicated small positives together as one system** | Never done. Candidates for new people: in-session fine-tuning, cheap form (+0.34); rest as a fourth class (+0.1 to +0.3 J); eye-net view (+0.28); mix exponent 0.9 (+0.2); PLS-DA view (+0.45 to +0.6 on two bases). For returning people: personal nets (+0.2), stronger stage (+0.15), template prior (+0.19), slot weight (+0.12). "Small levers do not stack" rests on one test of two of them | Low: mostly stored outputs |
| 2 | **Members with independent errors: redesign the teacher line** | The review's own estimate: an independent member at 70 % would add +3.1 for new people. The teacher (50 % alone) is the only one and already carries +1.4 J. Re-score the stronger teachers under the current chain and measure with the weight re-chosen; add cognitive-task heads (arithmetic-vs-rest, word-vs-rest) as a separate member; give the teacher its own session normalisation | Medium |
| 3 | **A weak-session policy** | P04-like sessions are detectable without labels and hold most of the loss; fine-tuning gains concentrate there (+1.6 to +2.1 per P04 session on 11 Nov) while strong sessions lose. Gate adaptation strength on session confidence | Low to medium |
| 4 | **Build one coherent redesign instead of add-ons** | A new core with the template stage and routing using its features. The pattern of this project: redesigned components gave the jumps, add-ons did not | High |
| 5 | **Use the last two uploads as designed measurements** | We do not know how our sealed score splits between new and returning people. Each remaining upload should answer a question as well as carry the best package. A self-check that fails loudly is free by rule | — |
| 6 | **Unlabelled PROTEUS data** | Explicitly allowed, never used: rest runs, the other seven-eighths of each trial, raw rate. Self-supervised pretraining of the nets, or rest-run galleries for the identity gate. Prior is modest: foundation models failed, but none was trained on this montage and these recordings | Medium to high |
| 7 | **Retest what was closed on a much weaker pipeline** | Sinkhorn balancing; MixStyle on SessionBN nets; EEGNeX in the trainable-filter slot; alignment inside the template stage | Low each |
| 8 | **Order robustness** | Not a gain: insurance. The rules no longer promise recording order, and we lose 3 to 5 points if it changes | Medium |
| 9 | **Spend the run time** | 38 minutes are free. Only fine-tuning uses them so far | — |
| 10 | **Whole systems, tested properly** | Each alternative system so far had one configuration. The transformer on synthetic sessions failed below chance on one seed, which looks like a bug, not a verdict | High, low prior |

**Needs your decision, not a run:** the cue generator of the Graz interface was searched for one family of random
generators only. Other families were not tried. If a search succeeded it would give labels for half the cells, which
is a protocol artefact and not decoding; the rules do not address it.

## 5. What the map does not show

- No axis hides an untested shortcut of several points that the record points to. The largest stated headroom is
  still information we lack: labels inside the session (+1.6 to +2.9) and more training people (+8 per doubling).
- The dev measure cannot rank changes below about +0.3, so "refuted" and "adopted" near that size are both weak.
- The record itself has holes: evidence from 27 Oct to 10 Nov lives in documents and untracked scratch only.
- The rules reader could not read the competition website (prize conditions, cohort description) or the private
  sealed bundle.

## 6. Questions worth asking the organisers

1. May a package ship labelled public data and refit on it together with the session's unlabelled windows at
   evaluation time? The FAQ says fitted statistics must use only the session's delivered windows; "do not train" was
   removed from the guide on 10 Nov.
2. Is the order of windows inside a sealed session still recording order?
3. Is using protocol regularities (cue order, class balance per run) acceptable?
4. Identity inference and personal models for represented participants: can the Discord answer be confirmed?
5. The audit: σ of what, over how many reruns; must external pretraining be rerun; what if the best score fails?
6. Which clock is the 60 minutes, and does a time-out end as "Failed" (no quota)?
