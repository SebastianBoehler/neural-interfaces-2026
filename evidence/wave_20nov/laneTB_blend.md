# Lane TB · Team B in hardcb_safe

- **Queued, 20 Nov 23:04 CET.** Real production folds and fresh full-session orders 71120–71123. Returning P04/P10 uses visits 2–3 with visit 1 enrolled.
- Leonardo refits the R1 seed-0 recipe because Galvani batch SSH rejected authentication. Mix doses are 15% and 25%, with nine members and a preselected three-member trim.
- v3 replay printed OK. Leonardo training smoke 64988026 completed. Fold fits 64988029 → real P01 mixer smoke 64988278 → streams 64988281 → collector 64988283.
- Conditional build and CPU blend-zero check are queued as 64988385. Accuracy, warm GPU cost, projected sealed time and the package decision are pending.
- All 12 R1 fits completed. Smoke matched the 190 pinned labels and both encoder oracles, then failed the bitwise posterior check at the first GPU fine-tuning update (max 4.034e-7). Fresh scoring stayed blocked.
- Retry 64990355 keeps exact pinned labels and exact pre-refit posteriors, with a 2e-6 bound and same-model repeats for the full GPU stream. Downstream jobs are 64990356–64990358.
