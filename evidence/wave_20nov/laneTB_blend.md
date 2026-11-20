# Lane TB · Team B in hardcb_safe

- **Queued, 20 Nov 23:04 CET.** Real production folds and fresh full-session orders 71120–71123. Returning P04/P10 uses visits 2–3 with visit 1 enrolled.
- Leonardo refits the R1 seed-0 recipe because Galvani batch SSH rejected authentication. Mix doses are 15% and 25%, with nine members and a preselected three-member trim.
- v3 replay printed OK. Leonardo training smoke 64988026 completed. Fold fits 64988029 → real P01 mixer smoke 64988278 → streams 64988281 → collector 64988283.
- Conditional build and CPU blend-zero check are queued as 64988385. Accuracy, warm GPU cost, projected sealed time and the package decision are pending.
- All 12 R1 fits completed. Smoke matched the 190 pinned labels and both encoder oracles, then failed the bitwise posterior check at the first GPU fine-tuning update (max 4.034e-7). Fresh scoring stayed blocked.
- Retry 64990355 measured pre-refit drift within the same hardcb_safe model (3.291e-9, exact labels). Smoke 64991339 checks exact mixer identity within each zero-arm call and retains a 2e-6 cross-run bound with exact labels. Scoring, collection and conditional packaging are jobs 64991340–64991342.
- Smoke 64991339 passed pinned labels and same-call identity. Its unchanged hardcb_safe repeat had max posterior Δ 0.021349 and zero label changes. Smoke 64992565 logs that diagnostic. Streams 64992566 add a 71120 baseline repeat across all folds. Collector and package are 64992567–64992568.
- Smoke 64992565 completed at 00:15 CET with all 190 pinned labels unchanged, exact same-call zero identity and live changes in all four positive arms.
- Warm A100 timing 64993397 completed. Hardcb added 33.93 ms/window on P01 (paired 34.06/33.80; components 31.99). Nine encoders added 7.82 ms/window and three added 3.54. Their 20,083-window projections are 3,394 s and 3,308 s before incremental load (+0.25/+0.34 s). Official limit 3,600 s.
- Fresh streams 64992566 started at 00:27–00:28 CET. Collector 64993398 waits on the streams, then conditional ZIP and CPU zero run in 64993399. Scores and the package decision are pending.
