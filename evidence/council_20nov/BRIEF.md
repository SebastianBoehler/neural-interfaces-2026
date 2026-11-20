# Council brief · Fri 20 Nov 2026, ~09:30 CET

Deadline Sat 21 Nov 23:59 CET. One sealed upload left; only the best sealed score counts; a Failed run costs no quota.
Best sealed: cap250 0.8299 (rank ~15). 3rd 0.8470, 1st ~0.859. Upload decision Sat ~14:00–16:00 with the user.

## Question

Reasoned from everything we measured, and checked against the challenge's rules: what could still move our sealed
score materially by Saturday afternoon, and what is the best final upload? Name breakthroughs we have not tried,
with mechanism, expected size, cost, and a decisive test that fits the time left. Disagree with the evidence or with
each other where warranted, and say where our measurements could be wrong.

## Read (all in the repo)

- Rules and bounds: `OFFICIAL_CONTRACT.md`, `evidence/audit_20nov/assumptions_rules_opus.md`.
- Audits and synthesis: `evidence/audit_20nov/*.md` (SYNTHESIS, first principles, sealed composition, data-path bugs,
  modelling assumptions, same-task people).
- Overnight results, `evidence/wave_20nov/`:
  - B5 learning curve: more people lift a member standalone +4.6 pp (N 5→9), the cap250 system −0.4 pp.
  - Lane V virtual subjects ≈ 0 or harmful; lane S Shin data −0.19 pp; B6 81 extra MI people +0.23 pp full system.
  - Clean-slate harness (`CLEANSLATE.md`): reference EEGNet system 44–46 %, cap250 73.7 % held-new.
  - Team A (newest FMs end to end) 58.8 %; team B (first-principles session model) 65.3 %;
    team C (ensemble + session gradient TTT) 69.7 % with TTT, 66.8 % without (T2 sweep, 62 recipes, plateau).
  - M1 learned in-context adaptation +2.3 pp over no context on a 59 % base; prefix BatchNorm alone 62.9 % beat it.
  - TTT inside cap250's final stack: +0.08 / +0.34 pp (intervals cross 0); heavy form exceeds 3,600 s.
  - T4: heavy TTT −3.0 pp on Scherer (drift); default TTT +4.5 pp on returning P04/P10 (two people).
  - Final fallback stack (cap250 + F-C25 hard + CBraMod): +0.39 pp held-new dev, ~+0.23 sealed-weighted, 3,211 s projected.
- Earlier: `evidence/review_19nov/forecast_calibration.md` (standalone gains shrink ~10× in the system; small dev
  increments transfer ~0.5× to sealed), `research_codex.md`, `docs/STATE.md`, `docs/LEDGER.md` (grep, 3k lines).

## Output

One file `evidence/council_20nov/<you>.md`, ≤150 lines: ranked recommendations with numbers, the decisive test for
each (runnable on Galvani/Leonardo before Sat 10:00), and your final-upload recommendation. Read-only otherwise;
commit only your file (`git pull --rebase --autostash && git push`, no AI author line, never force-push). Never upload.
