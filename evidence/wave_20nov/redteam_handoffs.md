# RT-1 · clean returning-fold handoff

**Status at 23:00 CET, Friday 20 November.** H4 complete. Leonardo has the H1 and H5 smoke jobs and dependent returning streams, H2, H3 and the hardcb_safe reference. The deadline for this handoff is 08:30 CET, Saturday 21 November. Candidate selection is pending the paired results.

## Instrument and checks

Real fold packages from `production-20261120-32a94ab7`, with P04 or P10 visit 1 enrolled and visits 2–3 held out. The real Solver emits one label per delivered window on full-session shuffled orders 71000–71003. The replay checks reproduced all six v3 reference scores and printed `OK`. The paired scorer independently regenerated the four pinned deliveries and checked full cell coverage. Data SHA-256 starts `2be4187fe33a`. Clean stock scores 70.20% P04 and 89.22% P10. Hardcb scores 70.06% and 89.22%.

Results use the official balanced accuracy averaged equally over participant × visit × interface cells. Changes are percentage points. Paired 80% intervals resample people then orders within people (20,000 draws, seed 20261121). Per-person intervals resample orders. **Two returning people is a thin sample.** The 20/80 returning mix and the 43% sealed returning share are unverified assumptions.

## H4 · same-source route suppression

Reused Lane RP jobs 64985932 (zero-change smoke) and 64986260 (full). The experiment held source weights fixed, disabled both recognition gates, and retained adaptation. All 16 stream-orders logged `recognised_at -1` and zero v6 errors. Each suppression tape matched its stock delivery row for row.

| Arm | P04 Δ [80%] vs stock | P10 Δ [80%] vs stock | Equal cells Δ [80%] | 20/80 Δ, unverified |
|---|---:|---:|---:|---:|
| noroute | −9.14 [−10.32, −7.97] | +0.50 [+0.17, +0.83] | −4.32 [−9.23, +0.55] | −1.43 |

P04 fell on all four orders. P10 rose on three. The registered criterion that suppression meets or beats stock on both people was not met. This contrast includes the adaptation that continues when recognition is suppressed. [Lane RP](laneRP_returning_person.md)

## Reproduction

Registration `050198b1`. Leonardo run `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/redteam-21nov`; release `15a639718601e959015fd533fa9106476614aa8c`. Job IDs and dependencies are in `jobs.tsv` in that run directory. Score the copied tapes with `python scripts/redteam/report_ret.py data/proteus_windows.npz RECEIPT_DIR`. [Dispatch](DISPATCH_21NOV.md) · [H1–H5 specification](redteam_fable.md#handoff-checks-for-an-executor-register-the-block-below-in-docsledgermd-before-scoring-anything)
