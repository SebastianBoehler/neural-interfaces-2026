# §318 CC30 × FBL2

Registered in `docs/LEDGER.md` before implementation/scoring. Existing §304 six-seed conditional-CORAL30 streams feed the original shuffled v6.6 chain. The new isolated patch keeps its original add/swap weights and applies the §299 factor two only to `sc_fbl`. It checks each joint knob against its own omission. Fresh SA/SS/FBL2 singles, zero controls and the joint arms receive their own count-filter and six CP outputs. The scorer retains the historical-control source/hash routing, archived route, pinned stream lock and complete-matrix guards. Conditional30 versus unconditional10 changes both alignment and strength; conditional10 versus unconditional10 is the matched conditioning test.

- Private Galvani snapshot: `/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117`.
- Snapshot source manifest SHA-256: `6cfdca10ad2801a7d8aee9755ea6e05eca67c12bd137b37ac64158374a80aca0`.
- Patched mix unit test: passed on the actual generated source, covering exact weights, add, swap, FBL2, both joints, and zero identity. Python source compilation passed.
- Real-data smoke array 3116564 runs zero, joint add, joint swap and FBL2; replay/guard job 3116565 and capacity-checked dispatcher 3116566 depend on it. Remaining 44 cells, at most four concurrent, and the complete scorer are released only after these checks pass. At submission 104 jobs were active and 13 running under the account limits.
- Earlier §304 recovery cells averaged approximately 4m44s in a three-cell smoke; this is a planning estimate, not a measured §318 duration. Queue delay, node load and preemption remain unknown.
- Selection is exploratory following §304/299. A stronger conditional30 than unconditional10 comparison changes both strength and conditioning. Only the prior conditional10/unconditional10 comparison matched strength. Returning-person, packaged gate and sealed-worker behavior are unvalidated.
