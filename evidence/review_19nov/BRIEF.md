# Review brief, Thursday 19 Nov 2026 (shared by all reviewers)

Repo: <repo> (Track 2 BCI decoding, Neural Interfaces 2026).
Read first: AGENTS.md, docs/STATE.md (long; skim section 1), OFFICIAL_CONTRACT.md, docs/LEDGER.md (index at top; 11k lines, grep it),
evidence/next_wave_16nov/{solution_space_review_17nov,scoring_audit_18nov,lunch_recommendation_18nov,recent_methods_18nov,FRIDAY_20NOV,EVENING_HANDOFF_18NOV}.md.
Apply the user's skills (read them): ~/Documents/GitHub/sebastianboehler/skills/{scientific-discovery/SKILL.md (+ references/coverage-audit.md, search-policy.md), intent-clarification/SKILL.md, engineering-workflow/SKILL.md}.

## Facts (verify where cheap)
- Task: streaming 3-class EEG decoding (MI / mental subtraction / word generation), input (1,41,480) @120 Hz, label-free
  adaptation inside predict allowed within a session; nothing crosses session boundaries; metric = balanced accuracy per
  participant/session/interface cell, averaged. Sealed delivery order is shuffled (not chronological). 3600 s limit on an A100 worker.
- Sealed best: v5.1b = 0.8198 (uploaded 12 Nov, 1,287 s runtime). User says we sit about rank 15. #5 was 0.841 on 17-18 Nov;
  #1 unknown here (Codabench API is bot-blocked from this machine; a mirror is https://neural-interfaces26.github.io/leaderboard.html#bci).
- 2 uploads left. Deadline Saturday; user wants the final submission decided by Friday evening.
- Staged but NOT uploaded since Wed noon: v6.6cap250 (runs/sealed/final_zips/v66_cp1_cap250.zip). Its dev gain vs v5.1b is
  about +1.1 pp public, +0.04 to +0.49 pp on nine external Scherer people (intervals include zero).
- Dev measure: 10 public people, shuffled orders, ~76 % FULL-SYSTEM DEV vs 82 % sealed. Recent increments are 0.0–0.3 pp.
- A large C++/CUDA "custom stack" exists (speed and bit-exact parity work). Compute: Galvani, Ferranti, Leonardo (on request), ML Cloud.

## The user's question
Why are we only ~rank 15 despite very broad experimentation, large compute/token spend and a custom end-to-end stack?
Are we thinking too narrowly or constraining ourselves artificially? What is the physical upper bound here, which state-of-the-art
methods (EEG foundation models, cross-subject/test-time adaptation, Riemannian/EA, ensembling, pseudo-labelling, data from
external corpora, etc.) transfer and have not really been run (or were run in too weak a form)? What would get us to top 3 / rank 1
by Friday evening with 2 uploads left?

## Rules for reviewers
- Read-only: do not edit tracked files, do not submit jobs, do not upload, do not commit. Only write your own report file.
- Cite evidence as file:line. Separate fact (measured, where) from inference and from proposal.
- Be blunt and concrete. Disagree with the existing documents where evidence supports it.

## CORRECTION (12:35 CET) — main is stale; the live work is on branch codex/friday-clean-stack
- Worktree: <campaign-checkout>/neural-interfaces-2026-friday
  (812 commits, ~77k lines since 18 Nov 13:30, not merged to main). Read friday/OVERNIGHT_TOP3.md, friday/DECISION_18NOV.md,
  friday/QUEUE.md, friday/SOLUTION_SPACE_18NOV.md, friday/RETROSPECTIVE_18NOV.md there.
- cap250 WAS uploaded on 18 Nov 18:45 CET: sealed 0.8298729992 (+1.01 pp over v5.1b; dev predicted about +1.1, so transfer was
  about 1:1 this time), runtime 2,554.9 s of 3,600 s. Rank #15 team / #16 participant. 3rd place 0.8470 (gap 1.72 pp).
  Top-10 cutoff 0.8351. Only 1 upload left (6 of 7 used). Public phase deadline shown as 21 Nov 23:59 CET.
- Overnight lanes on the branch (F-C13..C16, F17..F21) were mostly undecided or negative.
