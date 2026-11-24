# Constraint audit · what we lock in by our own thresholds (Opus review, Fri 20 Nov 2026 ~22:40 CET)

Read-only audit. No job, no upload. Rules come from upstream 2026-competition `main` @ `8447e181`, fetched inside the
submodule (pin `a66bc23` unchanged). Builds on, and does not repeat, `evidence/audit_20nov/assumptions_rules_opus.md`
(rules), `assumptions_model_codex.md` (constants) and `sealed_composition.md`. Written without reading the parallel
untracked `lock_audit_gpt.md`.

"Corrected instrument" means all three: shuffled delivery (full and within), the full current system (corrected chain
or real fold packages), intervals that resample people. **Arbitrary** means no rule and no measurement fixes the
number: it was chosen.

## Official rules that do bind (quoted, 8447e181)
- Runtime: "Sealed evaluations run on one NVIDIA RTX PRO 4500 GPU. The configured execution budget is … **60 minutes
  for Tracks 02–04**. Evaluations that exceed their execution limit time out." (`terms.md:42`)
- Quota and ranking: "A submission with final status `Failed` does not count." (`terms.md:25`); "Only the **best
  eligible sealed score** from you or your team is retained" (`terms.md:34`).
- Data: "Pre-train on any public dataset. Declare all of it … Public datasets may be used when their original terms
  permit competition use." (`terms.md:36-38`); "You may preprocess the public training data differently"
  (`scientific_faq_bci.md:100`).
- Causality: "a model may use the current and previously delivered windows, but no future window or label"
  (`scientific_faq_bci.md:95-96`); label-free adaptation "including statistics-based alignment or gradient updates
  inside `predict`" is allowed; nothing crosses the session boundary (`:121-127`); one integer of 3 classes per window (`:133`).
- Cohort: "Exact cohort size, identities, session allocation, and evaluation order remain private." (`:40`)

Everything below is ours.

## Sealed arithmetic that shapes the ranking
0.8299 = 0.566·N + 0.434·R (57/43 cell prior). With N at the dev held-new level (73.7 % full0, 78.3 % within0), the
returning cells must sit at R ≈ 0.95 / 0.89. Under the prior, returning is near ceiling (like P04/P10 at 90–97 %),
and about three quarters of the remaining error sits with new people. That pushes returning-only constraints down the
ranking. It does not clear the returning veto (row 27). That veto blocks new-person gains.

## Constraint table

### Training data and person set
| # | Constraint (where) | Origin | Set · instrument | What it excluded | Action · cheapest decisive test |
|---|---|---|---|---|---|
| 1 | **Every data or coverage lever was tested only by replacing the raw-FBCNet family** (MASS 5/6 or 1/6 of one 1/16 slot). The replay harness can only swap that slot (`laneH/full_replay.py:1,100`, `laneC/full_replay.py:29-51`), at full0 only | **Arbitrary** (instrument convenience inherited from F-C20) | 19–20 Nov · corrected, but full0 only and one slot | B5 coverage (`laneB5_curve.md:26`), B6 MI84, S Shin, V virtual, hard, CBraMod. All were applied to the weakest neural family (57.0 % alone). The stronger ones never got them: tangent core 65.6, Sinc 65.1, SBN-FB 64.0, FBLight 62.8, Teacher (5/16 of the mix). `CLEANSLATE.md:8-9` and `HANDOVER.md:23-28` generalise "member gains do not pass the chain" from this slot. Lane Z shows the loss happens at slot→mix (transmission 0.17), not in the stages (0.8–1.4) (`laneZ_chain_profile.md` §3) | **Retest.** Apply the B6 X pretraining recipe (+0.23 [+0.10,+0.37] at 5/6, `laneB6_mi84.md:55`) to the FBLight or Sinc six-seed family and swap it at its own slot, full0 + within0 in the B1/Z chain. Needs a family argument in `laneH/full_replay.py`. About 4–8 h on Galvani |
| 2 | Coverage lever "negligible" in the full system (B5: −0.07 [−0.14, 0.00]) | Measured, but only as row 1's diluted slot | 20 Nov · full0, one family | Any whole-system people-count lever, and the reading that coverage is the binding constraint (`SYNTHESIS.md` "binding constraint is cross-person coverage") | **Retest** after row 1. A whole-system N=7 vs N=9 run with the real fold pipeline (`production-20261120`) sizes it properly. Post-deadline |
| 3 | Sample-frequency fits: P04/P10 provide 43 % of source rows, and each of the eight gets a third of one of their volumes (`assumptions_model_codex.md` "Source fits") | Inherited default | Oct · F35 balance retest 19 Nov on the corrected chain: within0 −0.27, full0 +0.15, external tie | Person-balanced fits for the strong families | Keep (one form, undecided) |
| 4 | Scherer (the only other public 3-task cohort) stays out of member training so NI-V stays clean | Self-imposed | 20 Nov · Team C C3s pooled Scherer −4.28 [−5.66,−2.92] (`cleanslate_teamC.md:100-102`), one form, clean-slate base, 22/41 channels interpolated | Scherer as a low-weight auxiliary head on the 19 measured sites | Keep for the deadline |
| 5 | Teacher at 5/16 of the outer mix with 48.6 % alone. Its 38→3 map is fitted with recording-order EA (`assumptions_model_codex.md:20`) | Weights inherited from R19e/NXIF (recording order, P10-heavy) | Weights retested 20 Nov on the corrected chain: Z LOPO weights −0.24/−1.42, R1 nothing ≥ +0.3 nested | Retraining the Teacher itself with the 81-person corpora | Keep the weights (measured today). Teacher refit post-deadline |

### Window, crops and labels
| # | Constraint (where) | Origin | Set · instrument | What it excluded | Action · cheapest decisive test |
|---|---|---|---|---|---|
| 6 | Members train on the official 1–5 s crop. The trial lasts 7.6–8.0 s and feedback starts at 2.6–3.1 s (`crops347.md`); `scripts/export_windows.py:8-10` writes other starts | Inferred (match the test tensor). Rules allow any training-side preprocessing | Measured twice: §347 late 3.5–7.5 s crops −9.8/−15.0/−2.0 pp on 3 people (17 Nov, standalone, `crops347.md:138-145`); C1x five offsets −3.77 [−5.79,−1.69] (20 Nov, clean-slate, `cleanslate_teamC.md:44-48,89`). Small ±15-sample jitter is already in FT and hard | Near offsets (0.5–1.5 s starts) at low weight in strong families. They were never separated from the late offsets | Keep. Low-priority retest of 0.5/1.5 s only |
| 7 | **Fine-tuning learns from the inner v5.1b labels** (`v6_ft_labels=0`; `submissions/lean_v66/submission.py:1446,1562`). Member upgrades never reach the self-training teacher (`council_20nov/codex.md:24`) | Measured, stale | 14 Nov §206 (`git show a5843492:docs/LEDGER.md` lines 4768-4790): R−L −0.20 full / +0.10 within on the v6.1 chain. Instruments were STRICT-exempt and not people-resampled | Every later member gain (hard, CBraMod, TSEM) as a teacher signal. A teacher equal to the current pre-FT terminal was never built. The chain harnesses freeze FT labels (`laneZ_chain_profile.md:16`), so they cannot see this | **Retest.** The switch already exists: stream the 12 hardcb fold packages with `v6_ft_labels=1` on orders 71000–71003 against stored hardcb. About 2–3 h on Leonardo, tonight. Building a hardcb-mix teacher is the second form (new code) |
| 8 | Count-filter urn assumes the public block design (14 layouts, BH-first 0.75, Graz 36, capacities 12/10) | Inferred from public sessions | 20 Nov corrected: off = +0.04 full0 / −0.63 within0 (Z); S2 the same | Nothing under full shuffle. It pays only for within-run order | Keep |
| 9 | Three hard classes, plain CE. Run type (online/acquisition), interface and feedback phase are not training targets | Output fixed by rule; training targets inherited | Partly tested (§109 run-type statistics, §306 interface head), recording order | Auxiliary targets in strong families | Keep (low value) |

### Per-window causal design
| # | Constraint (where) | Origin | Set · instrument | What it excluded | Action · cheapest decisive test |
|---|---|---|---|---|---|
| 10 | Stages use the strict prefix (current window excluded) and fixed start ages: FT from 20 every 10 with 5 steps on seed 0 of 4 families; CSP from 20 every 20; contrast recognition from 60 (`assumptions_model_codex.md:36,41`) | Ages and doses inherited from recording-order chains; runtime | Light/heavy TTT inside hardcb, 20 Nov: +0.08/+0.34, undecided (`laneT_ttt_cap250.md`). §448 more updates −0.74 on Scherer | Putting the current window into rebalance/EM statistics (the rule allows it); stronger early-window priors | Keep for the upload. CPU retest on the stored chain (~1 h), prior ≤ +0.1 |
| 11 | FT cap 250 (`submission.py:1046-1049`) | Runtime (A100 cost), 15 Nov §268 | Never measured on accuracy: public sessions end at ≤199 windows | Little; sealed mean is about 189 windows | Keep |

### Recognition design
| # | Constraint (where) | Origin | Set · instrument | What it excluded | Action · cheapest decisive test |
|---|---|---|---|---|---|
| 12 | Thresholds route_tau 0.22 / bio_tau 0.26 at a 2 % session false-accept rate; galleries m=3 for P04/P10, m=1 for the eight (`assumptions_model_codex.md:36-37`) | Measured, stale | 6–11 Nov, mostly recording order and P04-driven. Scherer: 144/144 impostor routes once enrolled | Recalibration under shuffle; open-set gating (RP registered tonight, `docs/LEDGER.md:3286-3293`) | **Retest** (cheap). Count false and late routes in the stored held-new fold streams over 16 orders plus Scherer B-new. CPU, 30 min |
| 13 | The first positive ramp latches `recB` and permanently retires FT and CSP (`submission.py:1553-1555`) | Design of the v4.1/v5.1b era | R4 keep-adapt 20 Nov: NETBCI +3.68, P04/P10 −0.72/−0.10; R4b proxies missed the Spearman bar | Adaptation for returning visits far from ceiling | Keep for the upload (sealed arithmetic puts returning near ceiling). Undecided, not closed |

### Mixer and slots
| # | Constraint (where) | Origin | Set · instrument | What it excluded | Action · cheapest decisive test |
|---|---|---|---|---|---|
| 14 | Fixed outer weights and arithmetic pooling over 13 slots | Recording-order origin (STALE) | **Retested 20 Nov on the corrected chain**: Z ×2/×4/×8, LOPO weights and geometric pooling all ≤ 0; R1 nothing ≥ +0.3 nested; W and C2v not positive; Lane Lock still running | — | Keep (measured today) |
| 15 | A candidate replaces members inside an existing slot, so candidates compete (B6 X with hard: hcX −0.064, `laneP_stack.md:18`) | Arbitrary (F-C20 interface) | 19–20 Nov | Two data levers in two different families at once | Folded into row 1 |

### Runtime, size and parity
| # | Constraint (where) | Origin | Set · instrument | What it excluded | Action · cheapest decisive test |
|---|---|---|---|---|---|
| 16 | Projected time ≤ 0.85 × 3,600 = 3,060 s (`friday/qualify_package.py:26`) | **Arbitrary** (labelled SELF in `docs/STATE.md:14`) | 18 Nov | Superseded in practice | **Relax**: align it with row 17 so the script's `within_margin` verdict matches the rule actually used |
| 17 | ≤ 3,300 s (`final_stack.md:33`); ≤ 3,450 s (`laneR4_keepadapt.md:33`, `laneP2_package.md:33`) | **Arbitrary** margins. The 3,600 s budget is official | 19–20 Nov. The projection is 2,555 s × a 2080 Ti/A100 replay ratio; the RTX PRO 4500 was never timed | Unconditional R4 (3,929 s), more FT families or steps, B4 seeds, stock-protected FBLight (missed both RTX bounds) | Keep 3,300 s only for the final attempt: a late-Saturday timeout leaves no retry, and whether a timeout counts as "Failed" is unverified. Upload Saturday morning so that a 3,450 s candidate can be retried |
| 18 | fp64 FBCNet members on the GPU, TF32 off, one CPU thread (`submission.py:287-296,1708-1716`) | Parity engineering | Not measured on the RTX PRO 4500 (Blackwell, low FP64 rate) | Time that could buy FT and members (FT bypass costs 0.37–0.61 pp, Z) | Retest after the deadline. Not before the last upload (parity risk) |
| 19 | 90 MB package (`qualify_package.py:27`) | **Arbitrary**. No per-ZIP limit, 15 GB storage | 12 Nov | Larger ensembles; final_stack (90.5 MB) already overrode it | **Relax** |
| 20 | Posterior parity ≤ 5e-5 on RTX (1e-6 in smokes, 1e-3 for CBraMod) as an upload gate | **Arbitrary** as a gate; untouched stock exceeds 5e-5 on RTX | 16–19 Nov | Stock-protected FBLight's optimized archive (+0.38/+0.45 held-new) | **Relax** as a gate. Keep label equality on public fixtures as the bug guard |

### Families "closed" standalone or in a diluted slot
| # | Constraint (where) | Origin | Set · instrument | What it excluded | Action · cheapest decisive test |
|---|---|---|---|---|---|
| 21 | Foundation models: ten variants in the mixer, −0.23 to +0.26 | Measured at 1/6 of the raw slot; whole systems also measured (REVE −32.8, MANAS −15 pp) | 18–20 Nov | Nothing more for this deadline (CBraMod survived into hardcb) | Keep |
| 22 | "Output adaptation/ensembles exhausted"; interface detector "not worth building" (`review_19nov/strategy_opus.md:101-102`) | Recording-order closures (Oct; §103b per-interface rebalancing −0.41) | Ensembles retested 20 Nov (Z, R1, S2). Interface priors were never retested under full shuffle, where the two interfaces interleave | Per-cell (per-interface) rebalancing or EM priors | **Retest**: rebalance/EM with the oracle interface label on the stored full0/within0 chain. CPU 1–2 h. Close if the oracle gain is < +0.3 |
| 23 | Step 2 of the one-session returning lane was gated on "oracle headroom ≥ 2 pp" (`laneR_onesession.md:29,42`). The oracle was a per-person tangent model (44.6 %) against a 90–97 % stack | **Arbitrary** (the gate cannot fire with a weak oracle) | 19 Nov | The lever hunt for the eight one-session people (37.7 % of sealed cells under the prior) | Relax; covered by R4b and RP. Stack-size results are worth recording |
| 24 | Single-person experts stopped on an oracle ≥ +3 pp at 270 windows (`moonM2_neighbours.md:15,33`) | User stop rule (podium size) | 20 Nov | Smaller-dose and pooled-neighbour forms | Keep the stop. Label it "undecided below podium size" |
| 25 | Clean-slate systems 6–15 pp below | Measured whole vs whole | 20 Nov. Built in under 24 h against 5 weeks of cap250; the comparator is replayed cap250 on other orders | Nothing for the deadline | Keep for the deadline. Not evidence against ground-up designs in general |
| 26 | F-C22 (+0.54/+0.78 held-new) "did not qualify" on NI-V −0.017; FBL2 (+0.38/+0.49) vetoed by returning −0.21 and Scherer −0.02 (`docs/STATE.md:25-27`) | Self-imposed vetoes on noisy instruments | 19 Nov | Raw-slot multiresolution and FBLight doubling. Their interaction with hard is negative | Retest only within the fold packages next to hardcb, after the deadline |

### Decision weighting and gates
| # | Constraint (where) | Origin | Set · instrument | What it excluded | Action · cheapest decisive test |
|---|---|---|---|---|---|
| 27 | **Returning measured as 0.2 P04 + 0.8 P10, plus point gates** (returning ≥ −0.1 `laneP2_package.md:33`, `laneB1_final_choice.md:28`, `laneT_ttt_cap250.md:33`; P04 and P10 each ≥ −0.1 `laneR4b_dose.md:26`; ≥ −0.45 `laneD3_drift.md:39`; ≥ −0.3 `docs/LEDGER.md:3255`) | 0.2/0.8 was reverse-fitted on two sealed deltas from recognition-acting uploads (`assumptions_model_codex.md:74`). P04/P10 are about 13 % of returning visits (5.7 % of cells) | Lane A conditional replay with frozen banks; two people at 90–97 % | dsu_all (+0.34 held-new vs hardcb, −0.42 returning), FBL2, R4, D3, Lane A head dose | **Relax** to "report, do not veto": show 0.057 × P04/P10 + 0.377 × a stated assumption for the eight. dsu_all vs hardcb then comes out at +0.01 to +0.17 depending on the eight (CPU, stored outputs, minutes) |
| 28 | **Upload selection by do-no-harm conjunctions** (`final_stack.md:30-36`): held-new ≥ 0 in both orders, NI-V ≥ −0.3, runtime ≤ 3,300 s; follow-on rules add ≥ +0.2 over the final stack and returning ≥ −0.1 | **Arbitrary** under Force_Best (`terms.md:34`): 0.8299 is locked, so only E[max(0, Δ)] and the chance of ≥ +1.8 pp matter | 19 Nov 22:30 | With order noise ±0.3 and returning noise ±0.3, a true +0.3 held-new / 0 returning change passes all four checks with p ≈ 0.3 (0.75 × 0.63 × 0.93 × 0.63). Most small real positives are dropped, which runs against "stack small positives" (`AGENTS.md:21-23`) | **Relax.** Rank the candidates that exist by E[max(0, Δ)] with a runtime pass. Veto only a crash or a large measured loss (e.g. new < −0.5 or returning < −1.0). No new run needed |
| 29 | Tonight's bars: held-new +3.2 (`docs/LEDGER.md:3255`; `gap_audit_gpt.md:9`), returning +2.5 (`docs/LEDGER.md:3293`), +4.81 on the hard-return proxy (`gap_audit_gpt.md:11`), +1.0 (`cleanslate_teamB.md:26-27`, failed at upper +1.12), +2 pp by 22:00 (`moonM1_incontext.md:17`), +5 TA points (`laneQ_recog_hand.md:20`), Spearman ≥ 0.5 (`laneR4b_dose.md:26`) | **Arbitrary** with respect to mechanism size: podium arithmetic (1.81/0.566 = 3.2; 1.81/0.377 = 4.8) | 20 Nov | Mid-size effects (+0.3 to +1.5) in Team C selector, cluster-then-name, M1, R4b gating | Keep as go bars for a podium bet. Relabel anything below them "podium size refuted, stack size undecided" and feed it into row 28's stacking |
| 30 | 57/43 new/returning (`AGENTS.md:25-26`) | Inferred from withdrawn text; N = 20,083 fits 106 visits | 20 Nov | Mostly harmless: a 0.1 shift moves typical contrasts by ≤ 0.1 pp | Keep as a prior, with sensitivity |

## Top 10 by sealed score they could be hiding
Priors, not measurements: rough ranges in sealed pp, not additive.

1. **Row 1/2/15: data and coverage levers confined to the raw-FBCNet slot.** +0.1 to +0.5. **Arbitrary** (instrument). Test tonight: the B6 recipe in FBLight or Sinc at its own slot.
2. **Row 7: FT teacher is still v5.1b.** 0 to +0.5. Inherited and measured stale (14 Nov, old chain). Test tonight: `v6_ft_labels=1` fold packages (switch exists).
3. **Row 28: do-no-harm selection under Force_Best.** 0 to +0.3, upside larger. **Arbitrary.** Decide from stored numbers; no run needed.
4. **Row 27: P04/P10 20/80 returning veto.** 0 to +0.4. Half measured (reverse-fit on two points), wrong people. The sealed arithmetic (returning ≈ 0.89–0.95) caps the upside.
5. **Row 29: podium-size bars closing mid-size lanes.** 0 to +0.3. **Arbitrary.**
6. **Rows 17/18: runtime margins and fp64/TF32/one-thread parity choices.** 0 to +0.3 via more FT or members. The numbers 3,060/3,300/3,450 are **arbitrary**. After the deadline, except the margin choice.
7. **Rows 12/13: recognition thresholds (stale, P04, recording order) and the FT/CSP latch.** 0 to +0.3. Cheap stored-stream check.
8. **Row 22: per-interface priors closed in recording order only.** 0 to +0.2. Oracle check on CPU.
9. **Rows 20/26: parity and NI-V vetoes on FBLight and F-C22.** 0 to +0.2 (the interaction with hard eats most of it). **Arbitrary** gates.
10. **Rows 6/10: training-crop offsets and strict prefix (current window excluded).** 0 to +0.1. Crops measured negative twice; the current-window choice is ours.

Measured today and rightly kept: mixer weights and pooling (14), session-stage constants (S2), count filter (8), late
crops (6), clean-slate and foundation-model whole systems (21, 25), the 3,600 s budget and all causality and reset rules.
