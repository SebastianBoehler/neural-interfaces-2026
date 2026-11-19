# Project instructions

This project targets Track 2 BCI decoding in Neural Interfaces 2026.

- Read docs/STATE.md first (current state), then docs/LEDGER.md (what is closed) and OFFICIAL_CONTRACT.md.
- docs/STATE.md wins over any dated record. Old dated records live in git history, not in the tree.
- For rules, read upstream 2026-competition main, not the pinned submodule (pin is for code only).
- Pre-register every run in docs/LEDGER.md (mechanism, numeric prediction, refutation rule) before scoring it.
- Check ideas against the ledger index first. Score a part in the full current system under shuffled delivery, never
  alone; score a whole alternative system whole. The chain is the current system, not a boundary of the search.
- No gate or closure is final: before citing one, check what actually ran (code path, model class, alone or in the
  chain, which system version). State a rule before the run; say plainly when and why you go past it.
- A cleanup or review must also hunt silent no-ops: options parsed but not applied, overrides that drop a parent's
  behaviour, cache keys that omit a setting, runs whose code path differs from their description. Make them fail
  loudly.
- `python replay/check.py` must print OK before any chain result is trusted. It guards only the v3 chain in recording
  order; shuffled-delivery scorers need their own zero-change and pinned-order checks.
- Nothing may cross the session boundary in a package, module-level variables included (FAQ; review of 14 Nov).
- Rank 1 is not an upper bound (user, 13 Nov). Size explore work by its measured ceiling; an idea with a large
  ceiling gets several forms before it is set aside. Keep reporting the measured stack as it is.
- A bar inside the dev measure's noise (about ±0.5 across people; ±0.3 holds only across orders) does not close a
  direction: record "undecided". Bars are reading aids, not gates (user, 13 Nov): push and stack small consistent
  positives. Do not close on P10 or P04 alone; report both, neither decides. A new member or core is not closed until
  it was scored with the stages and routing using it.
- Intervals resample people as well as orders. Dev weighting like the sealed set (about 57 % new people; returning
  people about 0.2 P04 + 0.8 P10) is an UNVERIFIED assumption (cohort details are private; pending evidence/audit_20nov); report it next to J, labelled as such, and prefer the people-nested number for anything that was chosen.
- Keep compute busy (user, 12 Nov): submit questions as parallel waves, queue the next wave before the current one
  ends, and smoke-test every new script first. Compare run times warm and paired.
- Remove dead code, stale statements and narrowing assumptions as you go; git history is the archive. Before
  deleting a file, check that no queued job, collector or current document still uses it.
- You own this codebase: edit and clean up freely, commit and push often, with no AI author line in commits.
- Never edit submissions/*/submission.py in place: their SHA-256 values are pinned to uploaded or staged packages.
- Keep the official 2026-competition submodule pinned and unchanged.
- Keep our implementations and scripts outside the official submodule.
- Preserve the official metric and the model-facing tensor. Splits and training-side preprocessing are free (FAQ), and
  external people may serve as a second validation instrument, not only as training rows.
- Never commit datasets, environments, credentials or generated checkpoints.
- Track compact evidence with source revision, seed, metrics and validation limits.
- Validate locally before proposing a Codabench submission.
- Upload only with explicit user authorization; check live quota and phase first.
- Native CPU replay is not proof of Linux/A100 worker compatibility.
- Keep source files below approximately 300 lines and diffs surgical.
- Verify changes with the narrowest meaningful real-data check.
- Run local training only through `scripts/run_capped.sh` (one job at a time,
  process-tree RSS cap, MPS high watermark); never several MPS jobs in
  parallel. The Mac shares RAM with the GPU: parallel jobs froze it (Oct 22).
  Prefer Galvani for anything heavy.

## Threads, models and usage (user, 14 Nov; official Claude Code docs: sub-agents, advisor, costs, prompt-caching)

- Usage is the scarce resource: the weekly all-models limit is shared by every thread and subagent. Pace it to the reset.
- Models: execution threads and subagents on Sonnet (`.claude/settings.json` default; `executor` agent). Routine
  queue/log reads on Haiku (`cluster-check` only); `Explore` is overridden to Sonnet (built-in runs on Opus). Opus only for the research and
  coordinator threads, never for subagents. A Sonnet thread that needs hard calls gets `/advisor opus` in that thread
  only, instead of running on Opus. Not project-wide: each advisor call resends the whole transcript uncached, and on an
  Opus thread it duplicates the main model. Fable as advisor is not free; it counts against the plan limits (docs).
- Effort `medium` by default; `low` for checks; set model and effort at the start of a thread (switching models drops
  the prompt cache).
- Every turn re-reads the whole context, so long threads are the main cost. Start a fresh thread from a handover note
  once a thread passes about 250k tokens of context or its phase changes. The handover note has state, open jobs
  with ids and due times, and open decisions, in at most 80 lines. Do not keep 500k+ threads alive.
- Verbose work (log reading, sweeps, scoring runs) goes to a subagent so only its summary enters the thread.
  Subagents report in at most 15 lines.
- Idle waits: one background timer of about 50 min (inside the 1-hour cache TTL), never short polls.
