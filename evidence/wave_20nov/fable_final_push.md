# Fable final push · 21 November 2026, 20:10–20:30 CET

Read for this note: AGENTS.md, docs/STATE.md, DECISION_21NOV.md, fable_r8_scheme.md, sealed_scheme_inference.md,
final_review_21nov.md, sota_review_21nov.md, the outline of bottleneck_review_21nov.md, and PROPOSALS_2–9 in the
clean-room repo. Nothing was run, built or uploaded. No new accuracy number is in this note.

## 1. A change for tonight: none

**Upload em_half unchanged** (`833a28721b0de0fa…a4d`; local hash rechecked at 20:14, 43 members, 417 MB).

- **No untested lever fits the window.** A Scherer or PROTEUS check took 35–45 minutes today once its scripts existed,
  and a new package also needs zero-change, warm-up and timing pairs. Anything started at 20:15 would reach 21:15 with
  one instrument at best. Every output-side idea in reach already ran today and missed (soft balance −0.04, word
  consensus −0.03, terminal urn +0.04, early adaptation, Team B weight, covariance prior −0.09).
- **Quarter stays out.** It is the only measured positive left (+0.19 at 50/50 on PROTEUS, positive under all four
  schemes), and the sealed returning people are public PROTEUS people, which argues for the PROTEUS reading. Against
  it: the dose was chosen on the same two people, and nine Scherer people gave −0.67 full-session and −1.55 within-run
  on returning. A right call gains about 0.002 sealed, a wrong call costs 0.003–0.008. I would need about 75 %
  confidence in the PROTEUS reading and I have about 60 %.
- **An intermediate dose (0.375) only interpolates** between two instruments that disagree in sign. Not worth a run.

Two things to check by hand before the upload, neither needs compute:

1. **Storage and size.** This ZIP is 417 MB; the six earlier uploads were under 100 MB. The contract records no
   per-ZIP limit but a 15 GB profile store. Check free space on the Codabench profile first.
2. **Runtime is the one risk that cannot be retried.** The worst normal-node projection is about 3,280 s. On
   10–11 Nov a busy sealed worker ran route3geb 39 % slower (2,687 → 3,732 s) and it still finished. The same factor
   on em_half gives about 4,100–4,550 s, above the 3,600 s rule and near the 80-minute worker timeout. The package
   has a per-session refit limit and no global clock (removed on 14 Nov as cross-session state). I did not add one:
   an unqualified guard in the only upload is a worse risk than the one it covers. If a Failed run would hurt more
   than about 1 pp of expected score, the hedge is `hardcb cbgraph` (`6ba4092d…`), not a patch.

## 2. What I would push after the deadline

Ordered by how much each would change what gets built.

1. **Get the ground truth before the next experiment.** Ask the organisers for the delivery scheme, the per-cohort
   split of our six scores and the returning share, and read the top six write-ups. Six teams sit 3–4 pp above cap250.
   Our whole stack over `hss` is about +1.5 pp, so their difference is probably in the base model or the training
   data, not in the adaptation tail. Three weeks of tail work were steered by a scheme prior of 45/35/10/10.
2. **Rebuild the instrument around people, not orders.** Every returning decision in the last three days rested on
   P04 and P10, and their doses were chosen on them. The noise floor is about ±0.5 pp across people, so a 0.2 pp lever
   needs tens of people. Add multi-session external corpora as returning instruments (Scherer is one; Stieger 2021,
   Lee 2019 and Dreyer 2023 have more people and days), keep a lockbox of people that no selection ever touches, and
   record dev-to-sealed transfer for each upload. The order-loss fit to the 10 Nov rescoring needed a scale of
   0.14–0.33, which says dev effects of this kind arrived at a fraction of their size.
3. **Train the adaptation instead of stacking it.** The tail is rebalance → SessionEM → TSEM → gates → personal
   heads → own-space → CountFilter, each with a hand-set dose, and two instruments disagree on the sign of one dose.
   Meta-train one model on episodes that are shuffled unlabeled session prefixes, so the test-time procedure is the
   training objective and the dose is learned per session from evidence. In-context forms were started (moonM1,
   moonM2); I did not read their results, so check what actually ran before reusing or closing them.
4. **Make the base permutation-invariant over the prefix.** The base scores about 3.5 pp higher under within-run
   delivery than under a full shuffle, and the CountFilter latch fires in 139/144 within-run streams but 9/72 full
   ones. The system leans on temporal structure the organisers removed. A set encoder over delivered windows
   (transductive, order-free) should close part of that gap under full shuffle and cost nothing under within-run.
5. **Returning people: the day gap, not recognition.** An oracle that grants identity from delivery five gave
   +0.13 pp. P04 stays near 70 % with enrolment and carries 7 of 22 lost pp. The open problem is a personal model
   that survives a new day: enrol on all visits, train personal heads with across-day augmentation and
   day-conditional normalisation, and evaluate leave-one-day-out per person on the multi-session corpora above.
6. **One strong base against forty small actors.** Fine-tuned CBraMod and REVE at one sixth of a slot gave
   +0.2 to +0.26 held-new on an old system and were never scored as the base. Frozen features lost, in line with the
   October audit. Score a fine-tuned foundation encoder as a whole system (own alignment, own heads), with the
   runtime now spent on test-time refits moved to a larger offline-trained model.
7. **The +22 pp member oracle.** Learned stackers on posteriors failed twice, so the information that picks the
   right member is not in the posteriors. Test side signals instead (eye-position view, artefact state, the 56–57 %
   interface probe) as gating inputs, and train members for diversity on purpose. Lower prior than items 3–6.
8. **Late word misses** cost 3.71 joint pp on held-new and 76 % of the held-new loss comes after window 40. That
   is a class-separability problem in the base, not cold start. It belongs to items 3 and 6, with a word-specific
   error analysis first.

Process changes I would keep: whole-system comparisons, pre-registration and the scheme grid. What I would drop:
dose selection on two people, and packaging work on levers below the noise floor.

For the write-up: declare the Scherer corpus and its licence (cleared by the organisers on 11 Nov).
