# Lane A · Returning dose (S1 of strategy_opus.md)

Registered Thursday 19 Nov 2026 BEFORE any dosed arm was scored. Hypothesis: a recognised returning person is
under-weighted by cap250's RouteB personal head; raising its weight, or letting it dominate after recognition, gains
>= +1 pp on returning cells without touching new people.

## Mechanism and what "dose" means (read from submissions/lean_v66/submission.py)

`Route.step` (used by RouteB, 1/16 share): for a window routed to person j with ramp r,
`p = (mix - r a core1 + g r a h) / (1 - r a + g r a)`, a = core1's share inside the mix = `w0 (1 - w_rest)` = 0.0625
(= the "1/16" of LEDGER:712), g = 3, h = the person's personal LR head. The routed tail, TSEM prior (kappa 20 x r),
own-space stage (beta 0.1) and count filter then run on p.

Dose k multiplies only the head weight: `a_head = k a`; the removed core1 share stays a. Labels: 1/16 -> k=1 (stock,
bit-exact), 1/8 -> 2, 1/4 -> 4, 1/2 -> 8, 1 -> 16. Head mass in p at r=1: 0.167, 0.286, 0.444, 0.615, 0.762
(stock .. 1). "hard" = `p = (1 - r) mix + r h`: the head alone once the ramp is full. Only RouteB is dosed; the inner
v5.1b `Route` (gates, shadow, recognition) is untouched, so gates and recognition ages are dose-independent by
construction (asserted at run time). At r = 0 the code path is the stock expression, so a never-recognised session is
bitwise stock for every dose.

## Instruments

1. **Returning, conditional, 8 orders** (FULL-SYSTEM DEV, 2 people): the D-EArN native worker generalised from full0
   to the eight pinned orders (within0-3, full0-1, runs0-1). Source-fold packages `P04_s23`/`P10_s23` (visit 1 only
   of the owner enrolled; visits 2, 3 held), causal fresh EA banks substituted for the aligned FBC/Shallow members of
   that order, real raw/Sinc/FBL warm refits, count filter, cp. Cells: P04|2, P04|3, P10|2, P10|3. Control arms: the
   unmodified fold package (must equal the authenticated native-zero cell predictions for the order, bit for bit)
   and the dosed copy at k=1 (must equal it bit for bit). Limitation: production heads see 3 visits of a public
   person, this instrument one, so it understates the personal head.
2. **Held-new / false recognition**: dose acts only where r > 0. Evidence is the number of routed windows on held-new
   streams (LOPO receipts: 14 sessions x 8 orders, zero routed; Scherer: 0/36 streams, protected436) plus the Scherer
   NI-V arm rescored with the two best doses (identity expected, routed windows reported). No LOPO re-run.
3. **NI-V Scherer returning arm**: when it exists (evidence/wave_19nov/NIV.md), best two doses, 9 people.

## Prediction (numeric)

Equal-cell gain over stock (pp, mean of the 4 cells x 8 orders): k=2 +0.3, k=4 +0.7, k=8 +1.0, k=16 +0.9,
hard +0.4 (the first-visit-only head is weak and own-space/TSEM prior already inject personal evidence; larger doses
saturate then lose on P10, whose recognition is late, age 28). 80% interval about +-0.8. P(some dose >= +1.0 pp) ~ 40 %.
Recognition ages identical across doses; held-new / Scherer routed windows 0.

## Scoring and decision rule (fixed now)

Per (order, cell) balanced accuracy; gain = dose minus stock dosed copy. Report equal-cell, 20/80 P04/P10, each person.
Interval: orders resampled within person x people resampled (2 people, so descriptive; 80%), plus the 8-order paired
interval per person. Chosen dose = the dose maximising the 20/80 mean gain among doses with (a) positive mean gain on
BOTH persons, (b) >= 6 of 8 orders non-negative on the 20/80 gain; if the best is within 0.25 pp of a smaller dose that
also satisfies (a, b) take the smaller one (shrink toward stock). If none satisfies (a, b) the answer is "stock stays".

Lane claim: SUPPORTED if the chosen dose has equal-cell gain >= +1.0 pp and a positive 80% nested lower bound; REFUTED
in this form if no dose has equal-cell gain > +0.25 pp or every dose is negative on both persons; otherwise UNDECIDED
(do not close, per AGENTS: bars are inside the +-0.5 pp noise of a 2-person instrument). Held-new gate: any routed
window on a held-new/Scherer stream under stock must be reported with its score change; zero routed -> no harm.
