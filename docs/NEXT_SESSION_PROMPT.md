# Prompt for the next session

Continue our Track 2 (BCI decoding) entry for the Neural Interfaces 2026
sealed phase in
/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026. The goal is
**top three in the sealed phase**: the prize, and above all the sponsors'
internship and career opportunities, matter a lot to me. Ignore warm-up rank.
The sealed phase opens on 5 Nov: five uploads, one per day, best score
counts.

Read docs/HANDOFF.md first, then AGENTS.md, OFFICIAL_CONTRACT.md,
docs/SEALED_PLAN.md and docs/OPTIMIZATION_R10.md. Open older round docs only
for a specific number.

## First: finish what is running

The clean builds of R10, R10b and R10c from commit 5ee3adc are Galvani jobs
3017691–3017693.
1. Check harness agreement and the official replay.
2. Run the worker-image replica.
3. Stage the ZIPs in runs/upload/ and record them in SEALED_PLAN.md.

## Then: step back

Take a genuine first-principles look at the solution space before running
more sweeps. Every lever since R9 moved by ≤ 1 point, and our honest warm-up
proxy (0.536) sits about 20th among plausibly honest entries. The past step
changes all came from new information: alignment, the slow signal,
heterogeneous input streams and the balanced runs.

Challenge the ranked open directions in HANDOFF.md ("Solution space after
R10/R11"):
- more data with the same three tasks in one session;
- unified-label pretraining;
- self-supervised test-time training;
- better decision power from pooled seeds and participants;
- the seen-participant part of the sealed set.

Add anything better you find, including a literature and dataset search.
Give me a short plan ranked by expected sealed gain per day before spending
heavy compute. Use Galvani lanes generously once the plan is clear.

## Rules

**Evidence:**
- Select only on the mean of dev_session and dev_loso_all, and report
  dev_loso.
- Evaluate on all_loso (report P04/P09 separately) and seen_session only
  after a choice is fixed.
- Never tune on official, all_loso or seen_session.
- About ±1 point is noise: prefer plateaus and fresh-seed confirmation.

**Packaging:**
- Every package needs 100% harness agreement, an official replay with
  unchanged files and a worker-replica run before staging.
- Keep sealed runtime under about 30 min.

**Uploads and commits:**
- Never upload to Codabench without my explicit authorisation. Check live
  quota and phase first.
- Ask before committing unless I've approved it.
- Skip Scherer 2015. No teammates.

**Compute:**
- Run local training only via scripts/run_capped.sh, one job at a time.
  Prefer Galvani.
- Galvani 2FA: open a terminal tab running `ssh -fN mlcloud-galvani` and ask
  me to enter the code.
