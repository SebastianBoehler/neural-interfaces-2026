# Lane TB-grow · Team B on the hard-only base

Leonardo run `cap250/runs/laneTB-grow-21nov`. [Preregistration](../../docs/LEDGER.md#lane-tb-grow--dose-insertion-point-and-independent-sources-registered-21-nov-2026-before-scoring) · [Slurm dispatch](DISPATCH_21NOV.md)

## Instrument

Hard-only fold fixtures use `production-20261120-32a94ab7`. The ten held-new folds contribute 28 official cells per order. P04 and P10 contribute 8 returning cells per order, from visits 2–3 with visit 1 enrolled. Every arm uses the live hard Solver, with one EEG window at a time and a fresh copy per session. Local v3 replay printed OK. The PROTEUS export SHA-256 is `2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327`.

Orders 71140–71147 select by J, the equal mean of the held-new and returning cell deltas. The collector freezes one runtime-eligible arm. Orders 71148–71155 then replay the chosen arm, hard alone, and hard nine25. Paired 80% intervals resample people with their official cells, then orders within each sampled person. The 57/43 column assumes 57% new and 43% returning, with returning P04/P10 at 20/80. The sealed proportions remain unverified.

Timing pairs use the full Solver on the same warm A100, with order reversed on the second P01 stream. The projection starts from the previous 2,943 s A100 estimate for hard nine25, then adds the measured difference over 20,083 windows and any positive load and fresh-session setup differences. Official runtime on RTX PRO 4500 is unmeasured. Team B cost can grow with session length.

## Status

The Slurm chain is queued. Selection and held-out scores are pending.
