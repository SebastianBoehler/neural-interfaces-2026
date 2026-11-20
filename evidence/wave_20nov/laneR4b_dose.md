# Lane R4b · how strongly to keep adapting for returning people: dose arms and a confidence-gated form (registered Fri 20 Nov 2026 ~20:40 CET, before any R4b arm was scored)

Motivation: R4 (keep FT, CSP, lite5 TTT, head in the label after recognition) is +3.68 pp on NETBCI (far from ceiling, stock 68.6) and negative on P04/P10 (90-97 %). The coordinator's criteria
(`evidence/council_20nov/research2_sonnet.md` rank 1) are adopted as registered rules below next to my own. Code `laneR4b/` (builder `build_dose.py` imports `laneR4/build_keep.py`, untouched; `run.py` = B10 run with enrol/stream pairs and a per-window proxy log).
Release `/leonardo_work/AIFAC_P02_1675/ni26_laneR4b_20261120` (Leonardo, account 1675, all packages built from `hardcbB.zip` of the R4 release).

## Instruments (all vs hardcbB, paired per person, people-resampled 80 % intervals; NETBCI and ds007221g deltas are BA2 pp, P04/P10 BA pp)
1. NETBCI (B10 design, 19 people, enrol day 1, stream days 2-4). Dose arms and gated arms on 4 orders (full 72000/72001, within 72000/72001); hardcbB / R1a / R2 / R4 reference from the finished B10 run (8 orders, compared on the same 4 orders).
2. ds007221g (14 Graz people, enrol ses-01 stream ses-02 AND enrol ses-02 stream ses-01: two scenes = two environments; 4 classes mapped MI = feet/left/right -> task 0, rest -> placeholder tasks 1/2 alternately, scored as B10: BA2 of MI recall and rest specificity), 8 orders, hardcbB and all arms run fresh. The 48 ds007221lr (left/right only) people are NOT used: no rest class, so no non-MI gallery to enrol (stated).
3. P04/P10, Lane A instrument: ONLY stored tapes (Galvani, `ni26_laneR4_20261120/ret/out_*`, read only, R4-agent jobs untouched). New arms cannot be re-run there (CPU-SIF instrument on Galvani, other agent's queue). Gated forms are therefore scored on P04/P10 by a post-hoc switch between the stored hardcb tape and the stored R4 / R1a / R2 tape per window using the proxy computed from the hardcb tape (approximation: ignores that adaptation state differs). Dose arms: NOT delivered on P04/P10.

## Arms
Dose (R4 = (FT, CSP, lite5, head 0.5, refit every post-recognition grid point)): R4n = R4 without CSP; R4s2 / R4s4 = refit every 20 / 40 windows after recognition; R4t2 = 2 optimiser steps per refit (R4: 5); R4h1 = head weight 1.0 in the pseudo-label (R3 = weight 0, R4 = 0.5 are the other points of that dose).
Gated: R4 is computed in full; the gate opens its effect (FT members in the mix, CSP fused) per window iff >= 10 post-recognition windows exist and proxy (mean over the last 40 post-recognition windows, causal) < theta; closed = the final stack's own output. Proxies: `conf` (max of the system's own output posterior), `agree` (routed person's head and the population mix choose the same class), `ent` (1 - normalised entropy of the output, "prefix entropy"). `Gconf_on` is bitwise R4, `Gconf_off` bitwise hardcbB.
Silent-no-op guards: `Gconf_on` labels must equal the B10 R4 labels and `Gconf_off` the B10 hardcbB labels on the same stream (smoke, then every shared order); the proxy log length must equal the stream length; the proxy never reads a label; unrecognised sessions have gate irrelevant (bitwise stock).

## Threshold fit (rule fixed now)
Phase 1 runs `Gconf_off` (stock path, proxies logged) and `Gconf_on` on NETBCI and ds007221g. Per (person, stream) the proxy summary = mean over post-recognition windows 10..250; gain = BA2(R4) - BA2(hardcbB) on the same order. Step 1: Spearman(proxy, gain) per proxy, people-nested bootstrap (80 %). Step 2: leave-one-person-out theta on NETBCI that maximises the gated gain (rule: stream gated-on iff proxy < theta, gain = gain if on else 0); the fitted theta (full NETBCI) and two neighbours (+-0.03 on conf / agree / ent scale) are run in-system on NETBCI, ds007221g. P04/P10 get the same theta (not refit). Proxies with Spearman < 0.3 are recorded closed for these proxies only (coordinator rule), not for the idea.

## Predictions (pp over hardcbB, 80 % intervals)
- Dose, NETBCI BA2: R4 +3.68 (measured B10); R4h1 +3.8 [+2.0, +5.5]; R4n +2.8 [+1.2, +4.5]; R4s2 +3.0 [+1.5, +4.5]; R4s4 +2.3 [+1.0, +3.6]; R4t2 +3.0 [+1.5, +4.5]. Monotone in dose (R1a +0.36 < R4s4 < R4s2 < R4).
- ds007221g (stock probably 60-75, in-lab-like but new environment): R4 +2.0 [-1.0, +5.0], R1a +0.3 [-0.5, +1.0]; same ordering of arms as NETBCI.
- Proxy vs gain Spearman: conf 0.45 [0.1, 0.7], agree 0.40 [0.0, 0.7], ent 0.40 [0.0, 0.7]. Proxy level: NETBCI sessions conf 0.55-0.80, P04/P10 hardcb tapes 0.85-0.97.
- Gated (best proxy, fitted theta): NETBCI +2.8 [+1.2, +4.2] (about 75 % of R4's); ds007221g non-negative, +1.5 [-1, +4]; P04 / P10 post-hoc between -0.1 and +0.1.
## Rules (fixed now; bars are reading aids, P04 and P10 reported separately, neither decides)
Coordinator rule per proxy: Spearman >= 0.5 AND gated NETBCI >= +1.5 pp (80 % lower bound reported) AND P04 and P10 each >= -0.1 pp -> SUPPORTED, build into a full-chain run; Spearman < 0.3 -> closed for that proxy only; otherwise UNDECIDED.
Mine: a gated form is called "non-negative on P04/P10" if both point deltas >= -0.1; "keeps most of the far-from-ceiling gain" if its NETBCI gain >= 0.5 x R4's gain on the same orders. ds007221g reported alongside, not a gate. Sealed weighting (0.43 x ... UNVERIFIED) is not applied; no weight is assumed.

## Run log
Smoke 64979123 / 64979126 (NETBCI k0 full:72000 days 2-4, ds007221g k0 both directions): `Gconf_off` labels equal the B10 hardcbB labels and `Gconf_on` the B10 R4 labels bit for bit on all three days (0 of ~180 windows differ); proxy log length = stream length; gate open share 1.0 (on), 0.056 (off: only the pre-recognition windows). NETBCI proxies at k0: conf about 0.55 post-recognition (stored tapes: P04 0.75, P10 0.91).
Wave 1 (Fri 20 Nov 20:42 CET): NETBCI array 64979597 (Gconf_off, Gconf_on, R4n, R4s2, R4s4, R4t2, R4h1; 133 tasks, 4 orders), ds007221g array 64979598 (hardcbB, R1a, R2, R5, R3, Gconf_off, Gconf_on, R4n, R4s2, R4s4, R4t2, R4h1; 168 tasks, 8 orders).
