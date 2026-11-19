# Wave 19 Nov: parallel lanes toward the final upload (decision Friday 20 Nov 18:00 CET)

Context: evidence/review_19nov/{BRIEF,bugs_codex,strategy_codex,strategy_claude,strategy_opus}.md.
Sealed cap250 0.8299 (#15); 3rd 0.8470. One upload left. Decision Friday 18:00 CET.

Rules for every lane (shared checkout on main, several agents work in parallel):
- Follow AGENTS.md. Pre-register in your own card evidence/wave_19nov/<lane>.md (mechanism, numeric prediction,
  refutation rule) BEFORE scoring; add one index line to docs/LEDGER.md only when committing.
- Commit only your own paths (`git add <paths>; git commit -m ... -- <paths>`), then `git pull --rebase && git push`.
  If the git index is locked, wait and retry. No AI author line. Never edit submissions/*/submission.py (copy to a new
  submissions/<name>/ dir instead). Keep the official submodule untouched.
- Heavy work on Galvani/Ferranti (Slurm); eduVPN stays on; Leonardo is off. Smoke-test before arrays.
- Score full-system, every member re-streamed in delivery order, full-session shuffle, 8+ orders, paired vs exact cap250,
  people-and-order resampled intervals. Use the NI-V scorer when it lands (evidence/wave_19nov/NIV.md, being built by
  another agent); until then use the corrected Friday scorer and say which.
- Report gains labelled FULL-SYSTEM DEV / INDIVIDUAL-MODEL with cohort. Don't close a direction on one weak form.
- Final report in at most 15 lines: job ids, what ran, numbers, next decisive step.

## Network switch (user, 19 Nov afternoon)
Soon the user turns eduVPN OFF to use Leonardo. Then Galvani (and likely Ferranti) are unreachable from this Mac:
already-submitted Slurm jobs keep running, but you cannot submit, monitor or copy results until VPN is back.
So: get all Galvani/Ferranti submissions in NOW, with collectors running on the cluster (dependency jobs), writing
results to the cluster work dir. Prepare Leonardo variants of your next steps (see scripts/cloud/leo_*.sh and
memory notes in docs on Leonardo accounts; use compute nodes via Slurm, never the login node).
When your Galvani/Ferranti jobs (incl. on-cluster collectors) are submitted, append one line to
evidence/wave_19nov/QUEUED.md: `<lane> | <cluster> | <job ids> | <expected finish CET> | ready for VPN-off: yes/no`.
The coordinator tells the user to switch to Leonardo only when every lane says yes.
