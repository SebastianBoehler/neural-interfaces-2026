# v6 design: v5.1b plus fine-tuning, the eye-net view and the rest-class member on unrecognised sessions (13 Nov)

**Status (13 Nov, ledger §152): design only, no package before about 14 Nov 02:00 UTC.** The design that passes its bar
against v5.1b as uploaded applies three things **only on sessions the identity gate does not recognise**. Recognised
sessions stay exactly as in v5.1b. The three:
- in-session fine-tuning cum10 (warm start, 10 steps every 10 windows from window 30; the four SessionBN families with
  each base's own seeds; the k-5 traces of slot 9 frozen);
- the eye-net view in SessionEM;
- the rest-class shallow member at weight 1.
On all four augmentation draws: new +0.84 to +0.95 (without P10 +0.58 to +0.70), dev J +0.42 to +0.48. The cheaper gC5
schedule keeps about two thirds. The paragraphs below were written for the earlier "eye net and rest member
everywhere" form (§151: J +0.05 on the shipped members); that form is superseded.

Earlier status (§151): On v5.1b's own members the eye-net view and the rest member
add new +0.34 (+0.14 without the best person) and seen −0.24 (P04 −1.39, P10 +0.91), J +0.05. The large gains of
§140–§150 were measured against a study base (twins, personal nets, stage 0.15) about 0.7 below v5.1b on returning
people. Fine-tuning on the shipped members is the unmeasured piece (next GPU job when Leonardo is back).

No package is built before the research window ends (about 14 Nov 02:00 UTC), and no upload
without the user's go-ahead. This file collects what the measurements of 12–13 Nov say a next package should
contain, and what a build has to check. The numbers are in ledger §140–§151.

## What the measurements support

| Part | Package-form evidence | Decision |
|---|---|---|
| Eye-net view in SessionEM (OCU) | inside the stack on three seed sets (§140, §145 A); OCU's share +0.40 with nine eye-net seeds (§134) | **in** |
| Rest-class member (REST) | inside the stack on three seed sets; alone about 0 on two sets (§145 A); one shallow member at weight 1 is best (§144) | **in** |
| In-session fine-tuning (FT) on never-routed sessions | adds +0.47 inside the stack, of which +0.17 for the nine people other than P10 (§150); cheapest form gC5 replicates on one fresh set of two (§113) | **optional add-on**, only if its cheap form fits the run time |
| Personal nets for returning people (v5.2) | package form −0.36 / −0.40 seen, P04 −0.8 (§140, §105b) | **out** |
| Stage strength 0.15 | without personal nets about −0.1 (§119) | 0.10 unless §151's V6b says otherwise |
| Teacher variants | retrained control reproduces the stored teacher; the cognitive teacher gives +0.22 only inside the design input (§148) | unchanged teacher; m7u / m10 pending (§149) |

The v6 delta against v5.1b on the shipped members is §151 (V6): **J +0.05, below its bar.** The table's decisions hold
only relative to the study base. Before any build: (1) FT runs that start from the shipped SessionBN nets, scored on
v5.1b with OCU and REST; (2) the seen cost of OCU+REST for P04 explained or removed (it is −1.0 to −1.4 on the shipped
members).

## Components to build

1. **Eye net** (`studies/12nov/ocunet.py`, arm nomix): 7 derived ocular traces per window, each centred and scaled by its
   session's running statistics up to and including the window (causal, label-free), then a 1-D ResNet with
   channel attention and 3 outputs. For the package: train on all ten dev people, several seeds (dev used 3 to 9, all
   kept), trace with TorchScript, and average the seeds' posteriors. Its 3 log posteriors (clipped at 1e-3) are
   appended to SessionEM's 48 gaze and event features (GE view, 51 columns). Nothing else in SessionEM changes.
2. **Rest-class net** (`studies/11nov/rest4/rest4.py`, shallow, rest weight 1): a SessionBN ShallowConvNet trained
   4-way (MI / SUB / WORD / rest) on the task windows plus the 4 s before each Graz cue
   (`data/proteus_windows_start-4.5.npz` from `scripts/export_windows.py -4.5`). It is used 3-way, as an extra mix
   member at weight 1 in both tails, so every family's normalised weight shrinks by the same factor. Package form: a
   StreamBN trace like the other SessionBN members, several seeds averaged.
0. **Scope switch (§152):** the three changes act only while the session is unrecognised (no routed window so far),
   and the gate never sees them. §152 scored them with the gate fixed on v5.1b's shadow. The package form:
   - the shadow tail (the gate's input) stays exactly v5.1b's;
   - a second routed tail, "v6", carries FT's posteriors, the rest member and the eye-net view in its own SessionEM copy;
   - v5.1b's own routed tail keeps running beside it;
   - the output is the v6 tail until the first routed window. From that window the session uses v5.1b's routed tail,
     which has never seen a change, and fine-tuning stops.
   Recognised sessions are therefore exactly v5.1b from their first routed window, and windows before it carry v6. In dev
   no new-person session is routed, so this equals §152's V6F for new people and v5.1b for seen. Cost: one more copy of
   the rebalancing → SessionEM → TSEM tail (Route3.step about 19 ms per window on the A100 in v5.1b), and FT.
3. **FT (in, §152)**: cum10 (or gC5) on the four SessionBN families, never-routed sessions only, from window 30, with
   §120's five choices: SessionBN state after a refit, slots 6 and 9, no silent no-op, late recognition, schedule from
   the scored arms. Run time (§120 addendum 2, A100, paired with v5.1b): per never-routed session of 190 / 378 windows, cum10 adds 60.7 / 239.7 s for all 24 nets, 10.1 / 39.7 s for one net per family; gC5 31.1 / 122.2 and 5.2 / 20.4 s (v5.1b itself 15.5 / 31.6 s). The cost grows with the square of session length. Against a 45-min worst case (every session unrouted), only gC5 with one net per family fits at both lengths (29 / 35 min); a window cap would make it linear but is unscored.

## Checks before any upload

- The package replay equals the dev replay of V6 (§151) on the public sessions, per window, on CPU and GPU (as
  `check_v52.py --ref-schedule`).
- Both new members are present and load (assert at `load_model`).
- No silent fallback: a stderr line per session counts OCU features appended, REST posteriors used, and errors. Any
  error fails the replay check, not just the window.
- The eye-net standardisation is causal: the per-window output must not change when later windows are appended
  (check by streaming prefixes).
- Run time: a warm, paired A100 replay against v5.1b (the 3.9× cold start of 12 Nov is the reason).
- Worker image: Galvani, needs the user's 2FA.

## What is not known

- Sealed survival of new-people levers. The stage's dev-to-sealed ratio is about 0.2, and nothing new-people-specific
  has been uploaded.
- Whether REST's member gain holds with the shipped members. §151 answers this for dev.
