# Prompt for the next session

Continue optimising our Track 2 (BCI decoding) entry for the Neural
Interfaces 2026 sealed phase in
/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026. Goal: top
three in the sealed phase; ignore warm-up rank.

Start by reading docs/HANDOFF.md, then AGENTS.md, OFFICIAL_CONTRACT.md,
docs/STRATEGY_REVIEW.md and docs/SEALED_PLAN.md. Don't re-read the older
round docs unless you need a specific number.

Rules: select only on dev_loso and dev_session; evaluate on all_loso and
seen_session (report P04/P09 separately) only after a choice is fixed; never
tune on official, all_loso or seen_session. About ±1 point is noise; prefer
plateaus over seeds and weights. Every new package needs 100% window
agreement with the harness and an official ingestion and scoring replay
with unchanged files before it is staged in runs/upload/. Keep projected
sealed runtime under about 30 min (time on a Galvani A100 node). Never
upload to Codabench yourself; tell me which ZIP to upload and the expected
score. Ask before committing unless I've approved it.

Galvani needs my 2FA when the SSH session expires: open a terminal tab
running `ssh -fN mlcloud-galvani` and ask me to authenticate.

Next steps, in order:
1. Remind me of the two pending warm-up uploads (before 25 Oct 02:00 CEST)
   and record the results I report.
2. Collect the CBraMod fine-tuning results (Galvani array 2982784 and the
   Mac MPS run, see HANDOFF "Running experiments"). If it holds on ≥ 3 seeds
   and a weight plateau on both selection protocols, evaluate it, then
   improve the recipe (label smoothing, cosine LR, layer decay, LP→FT, seed
   weight averaging) and try REVE / ST-EEGFormer fine-tuned for diversity.
   Package the winner (plain-torch port or verified braindecode path).
3. Test a label-shift-aware causal reference (posterior-weighted,
   class-balanced running mean) aimed at the first 30 windows of a session.
4. Test selective information-maximising test-time adaptation (confidence
   ≥ 0.6 windows only, marginal-entropy term, reset per session).
5. Re-run the sealed rebuild from a clean commit, then finalise the
   sealed-phase upload plan and method description.
