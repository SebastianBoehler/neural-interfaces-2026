# Track 2 handoff · 17 November 2026, late CET

## Problem brief for a fresh methods session

Maximize official mean cell-balanced accuracy for motor imagery, mental subtraction and
word generation. Each call supplies one preprocessed `float32[1,41,480]` EEG window.
The model emits one immediate label. Delivery is nonchronological within participant-session.
The precise shuffle is private.
Only the current and previously delivered windows from that session can inform inference.
The model gets fresh state at each session. Sealed evaluation uses one A100 for 60 minutes.
Offline training is uncapped using permitted public labeled and unlabeled data.
The sealed deadline is 21 November at 23:59 CET. [Rules](https://neural-interfaces26.github.io/rules.html) · [Scientific FAQ](https://raw.githubusercontent.com/neural-interfaces26/2026-competition/main/codabench/pages/scientific_faq_bci.md)

Use this block alone for the first design pass. Derive alternatives from signal,
training supervision, session state and the scoring unit. For each, name the information
it uses, what could falsify it, and the cheapest experiment that changes a decision.
Then open the evidence below and remove duplicate or already-refuted forms.

## Artifact and run state

- Start with [STATE](../../docs/STATE.md), then read the relevant ledger sections and
  owning results. The [current coverage review](solution_space_review_17nov.md) links
  current evidence. The older solution-space map and 18:50 handoff are historical.
- Sealed best 81.9767% from v5.1b. Prepared v6.6cap250 is unuploaded, ZIP SHA-256
  `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`.
  The mirror shows 84.10% fifth entry. 83.70% is a provisional fifth-team proxy
  if the duplicate ETRI_EnSeption entries are one team. [Upload record](../../docs/SEALED_UPLOADS.md#log) · [Board](https://neural-interfaces26.github.io/leaderboard.html#bci)
- A100 C++ inference and adaptation match frozen on 3,833 public labels and all
  work counters. Wall saves 1.28% ordinary and 3.14% in the forced-adaptation pair.
  The image-specific overlay is ready for controlled experiments. Complete offline
  CUDA training parity and actual sealed wall time remain open. [Worker](native366_worker.md)
- Two independent short development replays overlapped in 108.62 s C++ and
  116.32 s frozen, with 142.50 s and 147.58 s serial and exact work. The CUDA
  trace identifies launch overhead, with no optimized CUDA speed measurement.
  Native early-refit skip is exact on DEV and saves 4.1% of the native
  personalization stage. The first fit at window 90 gained +0.0128 pp
  [80% −0.1079,+0.1350]. The 378-window schedule trial failed a label guard.
  The 190-window retry passed with unequal template warmup. [Evidence](native_iteration368_369.md)
- JS gains +0.2724 pp on exposed B2_pool new-person DEV, with a people/order
  interval spanning zero. Its conditional two-person returning result is
  −0.2879 pp. Cached branch choice has local oracle headroom versus JS
  (+1.33 pp B2_pool); fixed pooling and margin selection have no demonstrated
  lift. §371 found that the two largest ordered label disagreements deliver
  only 44.6% of the oracle gains. Blind switching in either loses accuracy.
  [Branch evidence](branch_ceiling362.md) · [Returning](js321_returning.md)
- Galvani showed no queued or running jobs for this account at 23:32 CET.
  Completed recent jobs 3122839, 3122845, 3122874, 3122875 are documented.
  The prior Gala automation was archived by the user.

## Working rules

- Keep frozen artifacts, the official submodule and unrelated working-tree changes intact.
  Use a fresh study directory for experiments. Upload requires explicit authorization and a
  live phase/quota check.
- Register numeric predictions, refutations and resource caps in the ledger before scoring.
  Check `replay/check.py` and the actual shuffled zero, prefix, order and build guards.
- Use the official equal-cell metric with paired people/order intervals. Report new people
  and genuine returning evidence separately, with P04 and P10 shown individually.
  The 57/43 weighting is a planning proxy. Public selection history limits generalization.
- Keep inference optimization bounded. An engineering result can improve iteration or
  budget without establishing a new score. A whole alternative earns its own full comparison.
