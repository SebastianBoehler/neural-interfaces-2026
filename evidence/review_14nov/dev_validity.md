# Can the dev measure be trusted for the next upload? (independent audit, 14 Nov 2026)

Scope: read-only audit of the dev measure behind the candidate "v6.2hL" (fine-tuning + eye view + rest member on
unrecognised sessions with labels from the baseline's own outputs; eye view, rest member and fine-tuning on the personal
route of recognised returning people; dev rows "v6f+ERF" in `runs/morning_table_0050.txt`, ledger §203 to §206).
Scratch scripts: `evidence/review_14nov/checks/` (`count_cells.py`, `count_runs.py`, `sealed_n.py`, `reweight.py`,
`loss_gain.py`). Nothing tracked was edited, no job was run. Official code was read at the pinned revision a66bc23 and,
with `git show` only, at the newest fetched revision bddaff3 (11 Nov); the pin was not moved.

## Short version

- The dev scorer computes the right cell quantity. It weights it for a population that is not the sealed one: P04 and
  P10 carry 60 % of dev J and 5.7 % of the sealed cells.
- The 0.41 to 0.46 "transfer" is a property of returning-people levers read on two people (it equals a P04 weight of
  about 0.22 instead of 0.5, the same on both uploads). It is not a measured property of new-people levers; for those
  there is no clean sealed reading at all.
- The new candidate does not have the old disease in the same form (P10 gains too, so the P04 weight moves the estimate
  by only 0.1), but its intervals are too narrow (they resample orders, not people), its returning half rests on two
  people after a search over about 30 arm and form combinations, and its new-people half has never been tested on the
  sealed set.
- New evidence found here: the logged 10-digit accuracies fix the sealed set at **20,083 windows = 105.9 public-size
  sessions**, which supports the 106-session cohort (60 new, 46 returning) and a mean sealed session of 189.5 windows.
- My estimate for the candidate: **sealed gain +0.65 points, 80 % interval +0.25 to +1.05, i.e. 0.826 (0.822 to
  0.830)**; about 95 %: 0.820 to 0.833.

## 1. Metric mismatch

- **Sealed metric** (`2026-competition/tracks/bci_decoding/objective.py:86-94`): cells = `stream_id` x `context_id`,
  balanced accuracy per cell (`benchmark_utils/metrics.py:45-61`, classes = union of true and predicted), plain mean
  over cells. Context is the interface of the run (`datasets/bci_studies.py:50-64`: AcquisitionBH and OnlineRawBH are
  BrainHero, AcquisitionGraz and OnlineRawGraz are Graz).
- **Dev scorer** (`replay/score.py:10-19`): balanced accuracy per `cell`, then the mean of a person's cells, then the
  mean over people. `cell` is session x interface: 28 cells for 14 sessions, and the window counts per cell equal the
  Graz and BrainHero run sums (`checks/count_cells.py`, `checks/count_runs.py`; e.g. P01: 96 = 36+30+30 Graz, 94 =
  34+30+30 BrainHero). So the interface is handled, and the cell quantity is the official one. `cell_ba` takes the
  classes from the true labels only (`score.py:11`); this equals the official form here because every public cell holds
  all three classes (29 to 35 windows each).
- **Weighting differs.** Dev "new" is person-equal over ten people (`score.py:18-19`), so P04 and P10 with six cells
  count like P01 with two. Dev J = 0.5 new + 0.5 seen (`score.py:32-35`, `morning_table.py:73-78`), with seen = the mean
  of P04 and P10. The sealed mean is cell-equal.
  - P04's data carry 0.5 x 1/10 + 0.5 x 1/2 = **30 % of dev J**, P10's another 30 %. On the sealed set each is 3 of 106
    sessions (**2.8 %**).
  - The eight single-session public people carry 40 % of dev J, all of it as *new* people. On the sealed set they are
    37.7 % of the cells, all of it as *returning* people, which dev never measures for them.
  - Sealed new people (56.6 %) are measured by dev's ten LOSO people at weight 50 %. For new people person-equal and
    cell-equal agree if every new person has six sessions.
- **Intervals.** `shufcombo.py:262-266` resamples the ten people (80 %, 4,000 draws). `morning_table.py:28-29,66`
  resamples **orders only**; people are the same ten in every order, so the table's intervals leave out the person
  term. Example (within, v6f, `checks/reweight.py`): orders only [+1.08, +1.40], people only [+0.92, +1.57], both
  [+0.87, +1.61]; 95 % both [+0.69, +1.83]. For returning people the table's interval is over orders of two people.

## 2. Population mismatch

- **Where 57 % / 43 % comes from.** The FAQ at the pinned revision says the sealed set "combines later sessions from the
  ten participants represented in the public release with ten participants absent from it"
  (`codabench/pages/scientific_faq_bci.md:25-26`), and the track page says "20-participant, six-session acquisition
  programme" (`competition_bci.md:61`). 20 x 6 = 120 sessions, minus 14 public = 106 sealed: 60 new (56.6 %), 46
  returning (43.4 %) = 8 x 5 + 2 x 3.
- **That text was withdrawn.** From upstream commit 4a8c943 (4 Nov) the FAQ reads "Exact cohort size, identities,
  session allocation, and evaluation order remain private", and the 20 x 6 sentence left the track page (checked with
  `git show` at 423b5af, 4a8c943, bddaff3).
- **Independent check (new).** `docs/SEALED_UPLOADS.md:300,427` logs plain accuracy to ten digits for uploads 4 and 5
  (0.8159139571, 0.8195488722). Both are k/N for the same N only at **N = 20,083** (16,386 and 16,459 correct;
  `checks/sealed_n.py`; the next solutions are multiples). Public sessions average 189.64 windows, so N is 105.9
  sessions. The total of 106 is therefore well supported, sessions are not truncated, and the mean sealed session is
  189.5 windows. The split 60 / 46 inside the 106 is still inferred from the withdrawn text.
- **Shares of sealed cells** (two cells per session assumed, as in all 14 public sessions):
  - new people: 60 sessions, **56.6 %**;
  - returning with three public sessions (P04, P10, sessions 4 to 6): 6 sessions, **5.7 %**;
  - returning with one public session (eight people, sessions 2 to 6): 40 sessions, **37.7 %**.
- **P04 is 3 of 46: correct**, and so is P10. By identity the right weights inside "returning" are P04 6.5 %, P10 6.5 %
  and 87 % for people with no dev measurement. The practical question is which dev person stands for the 87 %.
  Fitting the two uploads answers it: with returning share 0.434, sealed gain = 0.434 x (w x P04 + (1 − w) x P10) gives
  **w = 0.23 for upload 4** (P04 about +2.2, P10 about +0.13, sealed +0.267) and **w = 0.22 for upload 5** (P04 +3.51,
  P10 +0.07, sealed +0.361). Dev uses w = 0.5; "everyone else is like P10" (w = 0.065) under-predicts both by a factor
  2.3 to 2.8. So the defensible weighting is about **0.2 P04 + 0.8 P10**, fitted on two points.
- **No dev analogue at all:**
  - the eight single-session people as returning people (37.7 % of the sealed cells), in particular their sessions 4
    to 6 (dev tests one or two sessions after training, `experiments/protocol.py:134-137`);
  - P04 and P10 with three training sessions (the package) tested on sessions 4 to 6; dev trains on session 1 only;
  - the gate with three-session galleries (`SEALED_UPLOADS.md:273-275`: "never tested externally").

## 3. Selection effects

How much was looked at on the way to the candidate (ledger sections in brackets):
- **Fine-tuning forms:** cum10, gC5, delta1 / delta2 [§160], sweep 1 (16 forms, §194), sweep 2 (13, §197), sweep 3
  (12, §198), the stack forms (6, §198): about 45 forms on the same ten people.
- **Label sources:** recording-order chain, L, R, K, Q, T, blend, and the four of §145: about 10.
- **Eye view:** five refuted forms plus two averages [§156-§163], form A at three k, form B, two causal centrings, 9
  against 18 seeds [§188, §196, §177]: about 15.
- **Subsets of the combination** on six bases [§130-§152]: at least 6.
- **Returning-people arms:** E, R, ER, F, ERF x gC5, s15, s03, c01, e02 x two gate forms x four label sources
  [§186.4, §190, §198, §199, §201]: about 30 read on P04 and P10.
- **Orders:** 12 within, 8 full, 4 runs, up to 5 fine-tuning draws. **People subsets:** without P10, P04 alone, halves,
  126 splits. **Candidates compared in the table:** 6.

Gains measured on data that also chose them:
- **s15 over gC5:** chosen on within 0-3 (+0.20 form part), +0.11 to +0.14 on fresh orders, and **−0.00 [−0.23, +0.18]
  nested over people** on fresh orders [§203 text, sweepsel_fresh]. Fresh orders are not fresh people.
- **Labels R over L:** chosen on within 0-3 (+0.15), +0.03 on within 8-11, −0.2 on full [§198, §206]. v6.2hL uses L,
  so this does not enter the candidate.
- **The learning-rate effect** (the team's own catch): +0.29 in-sample, −0.00 on fresh orders [§194].
- **ERF s15 on the routed path:** picked on within 0/1 because P10 turned positive, then "firmed up" on further orders
  of the same two people [§198, §199]. More orders remove order noise, not person selection; with two people no nested
  estimate is possible.
- **The combination itself** (fine-tuning + eye view + rest member): its parts were chosen on the same ten people in
  recording order. The shuffled orders are new conditions, not new people. **No people-nested estimate of the main
  gain exists.** In its favour: ten of ten people are positive under within (per person +0.3 to +2.3,
  `checks/reweight.py`), it holds on all 24 orders, and the forms that were nested all came out near zero, which says
  the fine structure is noise but the main effect is not.
- **The last choice only, nested here** (`checks/nested_final.py`: pick among none / fallback / v6f / v6.1 on five
  people, score on the other five, 252 splits): within +1.42 held out against +1.44 in-sample, full +0.80 against
  +0.94, runs +1.08 against +1.14. The final pick among arms costs 0.02 to 0.14; the upstream design choices are not
  covered by this.
- **The candidate's exact new-people form has no row in the morning table.** v6.2hL fine-tunes with the s15 schedule
  on L labels (`build6.py ... --labels v51b --t0 20 --lr 1e-3`, ledger §206 build); the table's "v6f+ERF" row is gC5
  on L labels. §208 gives s15 + L against v6f as +0.24 within, −0.01 full, +0.05 runs, and the within figure is from
  within 0-3, the four orders that chose s15. So the team's J for the candidate (+1.09 in §208) contains about 0.1 that
  was measured on its own selection orders; on fresh orders s15 was worth +0.11 to +0.14 and nested over people 0.
- Where nested estimates exist they are: stage re-tune −0.02 / +0.16 against +0.44 in-sample [§166]; sweeps +0.11,
  −0.07, +0.10, −0.02, +0.21 then −0.00 [§194, §197, §198]; selector −0.02 to −0.06 against a +0.41 oracle [§193].
  In-sample minus nested is about 0.2 to 0.4 each time. That is the size of selection on this measure.

## 4. In-sample members

- Dev's returning protocol trains every member on all other people **plus the person's session 1** and tests on
  sessions 2-3 (`experiments/protocol.py:16-19,134-137`). So it does reflect that the package's nets saw the person,
  in the one-training-session form. That matches the eight single-session people in structure.
- Differences from the package:
  - sealed P04 and P10 have three sessions in training (dev: one). The baseline is then more personal than in dev, so
    additions probably add less there: dev is optimistic for these 5.7 %.
  - dev tests one or two sessions after training; sealed tests up to five later. The ledger's own diagnostic says most
    personal information is lost across sessions [§15], so dev is optimistic for personal parts on sessions 4 to 6.
  - new people: dev members saw nine people, the package's ten. Slightly pessimistic for the level, unknown for a
    paired gain.
  - the package's eye net and rest net are trained on all ten; the public replay (0.993) is in-sample and shows
    nothing (`docs/V6_BUILD.md:81-83`, listed by the team).
- Net direction for returning people: **inflation**, of unknown size, because every difference that has a sign points
  to less personal signal on the sealed sessions than in dev.

## 5. Shuffled delivery

- **Official code gives no evidence for any scheme.** The public evaluator still raises when a recording's windows are
  not in time order and when a stream or recording resumes (`benchmark_utils/streaming.py:95-121`), at the pin and at
  bddaff3 (the file is unchanged upstream). So the sealed backend is not the public code. The loader's order is
  `_acquisition_order` (`benchmark_utils/nb_task.py:296-328`): streams by first appearance, runs by BIDS run number,
  windows by onset.
- **What the documents say** (bddaff3 FAQ): one stream is one participant x session; "the same evaluation order is used
  for all submissions"; "State may persist across Graz and BrainHero runs in the same session"; the sentence "There is
  no shuffling within a stream" was removed. Organisers (13 Nov): chronology was removed "to avoid tricks ... of solving
  the track without using EEG", "with state reset between sessions".
- **Interleaving across sessions is excluded** by three things: the statements above; the public loop makes a fresh
  model whenever the stream id changes, so interleaving would reset state at every switch and cost every adaptive
  package the stages' whole value (about 9 points in dev), against an observed drop of 0.7 to 1.2; and N / 106 = 189.5
  windows per session, so sessions are neither split nor truncated.
- **What each scheme assumes:**
  - *within*: runs stay contiguous and in acquisition order, so the Graz and BrainHero blocks and the acquisition
    before online order survive; only the cue sequence inside a run is destroyed.
  - *runs*: runs contiguous, their order random, chronology inside. This is the only non-chronological order that
    passes the public evaluator's checks unchanged, but it leaves the cue sequence inside runs intact, so it does not
    remove a no-EEG trick. Least plausible alone.
  - *full*: no run structure; interfaces and acquisition / online runs interleave. The simplest thing to implement and
    the closest to "we removed chronology".
  - Not scored: a shuffle inside each interface block (cells contiguous), block shuffles, and mixtures.
- **Sealed evidence on severity.** Weighted to the sealed population, dev's loss of the v5.1b chain is **within −1.05**
  (0.566 x −1.46 + 0.434 x −0.52), runs −2.54, full −3.93 (new from `checks/reweight.py`, returning from §169). The
  rescoring cost the v3 chain package −1.05 and the TSEM package −1.17. At 1:1 transfer only within fits; full would
  need dev's order losses to transfer at 0.27.
- **The team's correction of §169 mixes two things.** It discounts the dev order loss by 0.41 to 0.46, the factor it
  attributes elsewhere to returning-people levers. An order loss that hits ten of ten new people is not such a lever.
  Either the factor is general, and then it also applies to the candidate's new-people gain, or it is not, and then
  the magnitude argument for a within-like stream stands.
- **Against within:** the ratio between packages (nxif −0.72 against TSEM −1.17) fits full better, and within needs a
  uniform offset of 0.35 to 0.5. Other teams' drops (0.4, 0.7, 1.4 on the board of 11 Nov) cap a uniform offset at
  about 0.4.
- **Verdict:** undetermined. Severity looks within-like or a little harsher; simplicity and the organisers' wording
  point to full. Dev's equal weighting of three schemes is not supported by either argument (runs is the least likely).

## 6. The transfer factor

Every upload with a dev figure recorded before it (`docs/SEALED_UPLOADS.md`):

| Upload | Lever | Dev figure before upload | Sealed paired gain | Ratio |
|---|---|---|---|---|
| 1, cand_nxif | first package | no paired sealed reference | 0.7795 | none |
| 2, TSEM package | all people | J +3.7 to +3.9 (lines 124-125) | +4.02; +3.57 after rescoring | 1.03 to 1.09; 0.92 to 0.96 |
| 3, routing v3 + gated early boost | returning + new | raw J +1.80 (`AUDIT_SBN.md:825-826`); "honest" +0.75 | +0.43; +0.55 after rescoring | raw 0.24 / 0.31; honest 0.57 / 0.73 |
| 4, v4.1 gate | returning | J +0.58 | +0.267 | 0.46 |
| 5, v5.1b own-space stage | returning | J +0.89 | +0.361 | 0.41 |

- **Whose property is 0.41 to 0.46?** Of the returning-people estimate. Splitting it: 0.87 comes from J's 50 % against
  the sealed 43.4 %, and about 0.5 from P04 counting half of "seen" while its response is 15 to 50 times P10's. With
  the weight 0.22 from section 2 both uploads are reproduced. The team's argument is right.
- **It says nothing about new-people levers.** The only all-people lever transferred at about 1 (upload 2, large). The
  only upload with a new-people small lever (upload 3, new +0.88 in dev) delivered 0.24 to 0.31 of its raw dev figure,
  confounded with routing; its permuted null alone was +0.30 (`AUDIT_SBN.md:822`).
- **The team's range "x 0.41 to 1.0" on raw J** leaves out upload 3's raw ratio. On raw dev J the three small
  increments gave 0.24 to 0.46, pooled 0.36. The 0.6 to 0.7 figure applies to an estimate already discounted before
  the upload.
- **Forecast record.** Stated centre against result: upload 2 +0.22, upload 3 −0.30, upload 4 −0.23, upload 5 +0.08
  (the one forecast built by re-weighting the population, PREDICTIONS_12NOV §63). Root mean square 0.22 points: no
  interval for this candidate should be narrower than about ±0.3 at 80 %.

## 7. What else makes the dev numbers optimistic or pessimistic (not in the team's lists)

1. **The table's intervals leave out people** (section 1). Stated [+1.07, +1.41] for new people under within; with
   people about [+0.87, +1.61].
2. **The sealed set has only about ten new people too.** With a between-person spread of 0.85 to 1.37 points in the
   gain, the sealed mean over ten new people scatters by 0.27 to 0.43 even if dev were exact. This term is in no
   estimate so far.
3. **The new-people gain is partly the baseline's own order loss.** Per person, gain and the baseline's loss under
   shuffle correlate at +0.87 (within) and +0.75 (full): gain = +0.67 + 0.39 x loss under within and −0.02 + 0.24 x
   loss under full (`checks/loss_gain.py`). In recording order the same form gains +0.79, mostly P10 and P01. If the
   sealed order costs the baseline less than dev's scheme says (full needs 0.27), the gain under that scheme shrinks
   with it. The robust figure across all readings is about +0.8 to +1.25.
4. **Scheme averaging with equal weights** includes runs, the least plausible scheme, and gives full one third,
   although severity evidence points to within (section 5).
5. **The wall-clock budget can remove fine-tuning from the last sessions.** v6.2g and later stop refits after 2,700 s
   (`docs/V6_BUILD.md:177-188`: "about the last 7 % of the stream" in the expected case). The worker has been 1.39
   times slower on an identical package (2,687 s to 3,732 s, `SEALED_UPLOADS.md:196`). Dev assumes fine-tuning on
   every session. Mean sealed session length is now known (189.5 windows), so the expected cost is the 190-window
   ratio, not the 378-window one. The clock also carries information across sessions, which is worth a look against
   the rule that nothing crosses a session boundary.
6. **Recognition rate.** Dev routes four of four returning sessions; externally the gate accepts 86.7 %. About six
   sealed returning sessions would take the new-people path instead, where §183 measured gains over the *unrouted*
   baseline only.
7. **Later sessions of new people.** Sealed new people bring sessions 1 to 6; in dev eight of ten new people have
   session 1 only.
8. **Pessimistic:** ten training people in the package against nine in dev; the Codabench board keeps the best score,
   so the downside of an upload is the slot, not the rank.

## Estimate for v6.2hL

Dev inputs (`runs/morning_table_0500.txt`, rows v6f+ERF; `checks/reweight.py`): new +1.24 / +0.94 / +1.00 (within /
full / runs) with gC5, and +1.48 / +0.93 / +1.05 with the package's s15 schedule (§208); +0.79 in recording order;
returning +0.89 / +1.18 / +0.82 with P04 +1.14 and P10 +0.64 under within.
Face value on the sealed population (56.6 % / 43.4 %, P04 weight 0.22): +1.03 / +1.00 / +0.87 with gC5, about
+1.17 / +1.00 / +0.90 with s15.

- **New part:** dev about +1.05 to +1.1; selection and transfer 0.6 to 1.0 (centre 0.75: robust main effect, parts chosen on
  the same ten people, no sealed precedent) → +0.8 sealed for new people, x 0.566 = **+0.45**, standard deviation
  about 0.30 (dev people 0.29, sealed people 0.35, scheme 0.2, transfer 0.17, all x 0.566).
- **Returning part:** 0.22 x 1.14 + 0.78 x 0.64 = +0.75; two people and a 30-way search → x 0.7 = +0.55; x 0.434 =
  **+0.24**, standard deviation about 0.22.
- **Budget and recognition:** about −0.04.
- **Total: +0.65 points, standard deviation 0.37.**
  - **80 % interval: +0.25 to +1.05 → 0.8263 (0.822 to 0.830).**
  - About 95 %: +0.0 to +1.3 → 0.820 to 0.833.
  - Chance of at least +0.6 (third by team on the board of 13 Nov): about 55 %. Chance of under +0.1: about 7 %.
- The team's "+1.0 to +1.2 x 0.41 to 1.0" (+0.41 to +1.2) has about the right upper half; its lower end is too high by
  about 0.15, and its centre by about 0.15.

## Three changes that would most improve the dev measure

1. **Weight dev like the sealed set and bootstrap people.** Report 0.566 x new + 0.434 x returning, with returning =
   0.2 x P04 + 0.8 x P10 (the weight both uploads imply) next to P04 and P10 separately, and give every figure a
   two-way interval over people and orders. Add the folds dev can still build for returning people: train on sessions
   1-2 and test on 3 (the three-session case), and session 1 → 3 against 1 → 2 (decay with distance).
2. **Make the headline the nested number.** For every chosen part (form, label source, routed arm, candidate) select
   on five people and score on the other five, over orders not used to choose, and quote that. Where it cannot be
   nested (two returning people), say so and apply the measured in-sample minus nested gap (0.2 to 0.4 points).
3. **Calibrate the order scheme on the sealed rescoring instead of averaging three schemes.** Weight schemes by how
   well the population-weighted dev loss of the rescored packages reproduces −0.72, −1.17 and −1.05; add an
   interface-blocked shuffle; drop or down-weight runs; and report each candidate's gain against the baseline's order
   loss, so the part that depends on the scheme is visible. Ask the organisers whether runs stay contiguous.
