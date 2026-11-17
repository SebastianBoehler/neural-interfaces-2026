# NI26 thread retrospective — 17 November, 13:00 CET

This records the campaign since the Claude handover, including overnight work.
It separates measured improvements, diagnostic findings, unsuccessful methods and pending work.
STATE and the owning handoffs remain the live entry points after this snapshot.

## Shared assessment

We have a small selected full-system DEV improvement, useful mechanism evidence and validated CPU speedups.
We have not verified a new sealed score or a top-five-ready submission in this campaign.
The strongest recent individual-model gains have repeatedly failed transfer to other people or the ensemble.
New ideas remain permitted; the current ensemble and representation are not the search boundary.

## Actual achievements and their limits

| Item | Measured outcome | What it establishes |
| --- | --- | --- |
| Selected CC30/FBL combination, §318 | Six-seed B2_pool new-person FULL-SYSTEM DEV76.2793→76.5517%, +.2724percentagepoints | Best measured selected system change; people/order80%interval−.007→+.546pp,6/10people improve; no sealed/returning confirmation |
| Native classical inference, §301 | Paired CPU component113–117→63–64ms/window | About1.8× on that component; not whole-worker latency |
| CPU head repair, §343 | Same-node warm sklearn19.7467→16.4362s; CPU117.3624→93.3867coreseconds | 16.76% less wall time,20.43% less CPU time at reference optimizer/stopping quality |
| Independent head fold, §343 | sklearn15.4332→10.7565s | Same-direction improvement on a second source fold, single cold pair |
| Head prediction parity, §343 | 1,534 predictions:767unique held windows across two people/two delivery orders; zero label changes | Preserved predictions in these complete streams; max probability error6.65e−8, no held accuracy read |
| Head memory, §343 | Extra sampled fitting RSS77,796→71,748KiB | 7.77% smaller increment in single fresh-process probes; loaded baselines differed, so no general absolute-memory claim |
| Candidate fidelity, §321 | Genuine primary/inner source fits, original-class exports, causal views, reset/deepcopy and actual stage/FT guards | A much better audited experimental artifact; remaining comparison is conditional on retained production priors |
| Research workflow | Reusable scientific-discovery skill, explicit registrations, fresh-context workers and scheduled primary-source blocks | Better process and reproducibility; not evidence that the entire solution space was exhausted |

§343 uses custom C++ value/gradient kernels and the existing SciPy L-BFGS-B optimizer.
Neural-network training still uses PyTorch. This is not a complete C++ neural training framework.
The historical243s solve was not reproduced: current old-native repeats were about40–47s with tighter stopping.
The accepted repair is compared contemporaneously against sklearn, not advertised as243→16seconds.
All useful source/evidence checkpoints are pushed. Frozen submissions and the official submodule remain unchanged.

## Findings that changed our understanding

1. **Individual improvement is not ensemble improvement.** Session variance adaptation accounts for roughly4
   INDIVIDUAL-MODEL percentage points in the tested raw decoders. Mean-only changes do little. The gain survives
   seed averaging but does not reliably improve the full system, even after source-only calibration (§339).
   About84–85% of the new member's corrections overlap ensemble successes. That is descriptive evidence;
   it does not by itself prove the cause. The attenuation is already visible before the final CP stage.
2. **Better reconstruction does not imply better decoding.** Concurrent covariance supervision (§344) improves
   the final linear covariance probe but reduces exposedP09 individual accuracy64.061→61.175%.
   Saved checkpoint trajectories are mixed, so a blanket catastrophic-forgetting explanation is unsupported.
3. **The teacher must be measured in the actual split.** Honest source-person-crossfitted distillation (§345)
   gives61.665% versus raw64.061%; the matched teacher is63.382%, not its historical ten-person69% figure.
   Its uncertainty targets were not useful in this pilot. This does not close all distillation methods.
4. **A single-person positive needs a frozen follow-up.** Direct slow/high-band summaries (§346) give
   P0964.061→65.447%, +1.386INDIVIDUAL-MODELpp. Without refitting, the same method loses1.482pp onP02
   and4.837pp onP03 against each person's actual A6 reference. It is not ready for system expansion.
5. **Weak-comparator gains are easy to misread.** Retaining spatial Atom tokens gives44.630→51.347% onP09,
   but remains below the stronger raw64.061% reference. Similar caveats apply to visual priors and pretexts.
6. **Parity must include held inputs and the actual stopping rule.** The first fast source-span solver passed
   training checks yet changed held predictions. Matching the reference optimizer/stopping and checking full
   streams fixed that. Faster convergence to a different numerical solution was not accepted as a faithful port.
7. **Inspect actual code, data and execution.** A configuration option is not an executed experiment. Frequency
   attenuation is not mathematical information absence. Outside-crop seconds were not globally unused:
   older ordinary-BN EA crop training and §277SSL already consumed some of them.

Actual exploration included covariance/spectral features, raw CNNs, state-space models, microstate summaries,
visual and EEG pretrained representations, generative session inference, normalization and learning objectives.
Coverage is broader than one representation, but many alternatives remain limited pilots or posterior additions.
These negative forms do not establish universal architecture ceilings or exhaustive coverage.

## Process mistakes and changes

- Some new validation wrappers had incorrect metadata cardinalities, device assumptions or export expectations.
  Small-prefix timing also underestimated full preparation costs. These caused failed/cancelled work and delay.
- This was not proof that Claude's official metric or earlier evaluator was wrong. The official metric was retained.
- The candidate audit expanded farther than useful for a quick update. Its retained global priors, old calibration
  and auxiliary weights prevent calling the resulting comparison independently held-out generalization.
- Repeated orders/seeds are not independent people. Reused DEV selection is not fresh confirmation.
- From here, every proposed gain keeps its comparator, cohort, measurement level and source scope attached.
  Full workload cost checks precede expansion; failed runs remain visible; no weak positive is silently promoted.
- Fresh workers retained canonical history. We did not perform blind cross-provider replications.

## Queue and next decisions at this snapshot

| Work | Actual status | Next decision/time |
| --- | --- | --- |
| §321 returning comparison | Pilot3119222_0 passed12m56;8CPU cells3119252 running,7pending; collector3119253 waiting | Finish BOTHpeople/all8orders before any metric; approximately13:20–13:30 if remaining cases resemble the pilot, later if recognition/FT is slower |
| §347 late-crop source qualification | First preparation failed a source posterior guard; diagnostic3119299 isolated rounding-order difference and restored bitwise source-prefix outputs | Prepare a source-only retry, then qualify ALL source crops before any training; no accuracy ETA yet |
| §347 possible training | Not queued | Only after qualification/cost proof: original raw reference, extra official repeats, and matched late-crop updates |
| Primary-source research |10:49:35–11:04:39CET block complete15m04 | Next scheduled block about14:51CET; not replaced by intervening audits |
| Submission decision | Provisional18:00CET | Assess full-system evidence, artifact fidelity and runtime; upload remains the user's decision |

§321 is CONDITIONAL ENGINEERING: fixed production MAD/teacher geometry/old OOF calibration/rest-eye auxiliary
weights retain exposure. A positive can inform this artifact's behavior, not prove a new rank or independent gain.
The pilot showed actual recognition/FT, original inner/off identities and unchanged model buffers.

For §347, source timing has1,894qualified end markers and4missing endpoints among1,898cues.
True cue lengths span7.634–8.080s. Fixed3.5–7.5s crops fit the qualified trials; they overlap the official1–5s
crop by1.5s and add2.5s outside it. The four unsupported extras are omitted in BOTH matched extra-data arms;
their original official rows remain. No padding, per-trial offset adjustment or extra labels are assumed.

## Submission and rank

Last verified uploaded score: v5.1b .8197668759. Prepared v6.6cap250 remains unchanged and unuploaded here.
Its ZIP SHA256 is9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed.
Last verified board at01:12CET:16thentry/15thteam; later bot checks prevent a current verified rank.
Earlier projected rank ranges were unvalidated scenarios, not achievements. Top-five readiness remains unverified.

Primary internal evidence: [state](../../docs/STATE.md), [queue handoff](OVERNIGHT_16NOV.md),
[solver343](solver343.md), [covariance344](retention344.md), [distillation345](distillation345.md),
[direct summaries346](direct346.md), [candidate321](js321_returning.md), [research1051](research_4h_17nov_1051.md).
