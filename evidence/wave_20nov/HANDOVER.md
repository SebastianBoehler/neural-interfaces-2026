# Handover · Fri 20 Nov ~15:20 CET (Mac offline ~3 h)

Cluster jobs keep running; local agents and the hourly keeper stop while the Mac is offline. On return: check ssh
(Galvani 2FA; Leonardo `step ssh login`, cert expires 22:39; Leonardo SSH sometimes times out under eduVPN — retry),
read results below, restart agents.

## Running / queued unattended (collectors chained afterok)
- Galvani (~36 running, ~70 pending): R4 keep-adaptation returning arrays (ni26-R4-ret, -niv, -cp), lane P2 returning
  collect (ni26-laneP2-retcollect), NI-V collectors (ni26-niv-collect/-ret/-enrol), B1 sparse20 returning (ret-sparse20).
- Leonardo: R4 NI-V (r4-niv, ~53 running), B10 one-session returning stress (b10-run), cap250 fold waves
  (wave 1 had 6 failed `cap250-w01-014-worker`; a fix agent was started 15:20 to re-chain waves 2–4 — check the
  card `cap250_pipeline.md` for whether it finished before the Mac went offline).

## Results already in (20 Nov)
- Lane P best joint arm ds_all (hardcb + lite5 + FBL2 + dispersion correction, per-session reset): +0.86 pp vs cap250;
  B7 seed-robust pooled +0.72 [+0.11, +1.26] (13/13 sets positive); vs final stack +0.38 [−0.11, +0.81].
- Lane Z: the absorber is mixer dilution + redundancy; later stages pass gains. R1: mixer re-weighting doesn't help.
- Lane N (NI-V 8 orders): finalists ≈ cap250 on Scherer; light/heavy TTT −0.3 to −0.5 on within orders.
- Final stack runtime ~3,130–3,150 s projected (paired timing).

## Waiting for an agent (stops while offline)
- P2: packaging ds_all with a real member (not the stand-in), with/without FBL2; checks chained — read laneP2_package.md.
- R4: keep FT/CSP/TTT after recognition — results land in laneR4_keepadapt.md / Galvani release.
- B10: laneB10_returning_stress.md.

## Decision Sat ~10:00
Fallback: final stack hardcb (sha 958fd3f3…). Upgrade to a packaged ds_all (or R4 form) only if its package passes
zero-change, NI-V shows no external harm (watch FBL2), returning not worse, and runtime projects ≤ 3,450 s.
Upload Sat ~11:00–14:00 with the user's explicit go-ahead.
