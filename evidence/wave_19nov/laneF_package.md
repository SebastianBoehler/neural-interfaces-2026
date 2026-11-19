# Lane F · Final package: facts and one-command qualification

19 November 2026. No upload, no Codabench action. Tool: `friday/qualify_package.py` (236 lines).

## 1. Facts

| Fact | Value | Status | Source |
|---|---|---|---|
| Time limit, pinned YAML | 3,600 s (`execution_time_limit` for warm-up and sealed) | **verified** (pinned submodule `a66bc23`) | `2026-competition/codabench/competition_bci.yaml` lines 62, 73 |
| Time limit, live sealed | raised by organisers; number unknown. 4,800 s is one participant's report | **repo-claimed** (organiser A. Caillet, Discord 18 Nov 07:50 CET); number **unknown** | `OFFICIAL_CONTRACT.md` L71–76, `friday/DECISION_18NOV.md` L64 |
| Planning budget | 3,600 s; the script requires projection ≤ 0.85 × 3,600 = 3,060 s | our rule | `bugs_codex.md` §4 |
| Sealed worker GPU | slower RTX, model **unknown**. Public rules still list one A100 / 60 min | **repo-claimed** (RTX), model **unknown** | same; `docs/COMPETITION.md` L93. Our proxy is a Galvani RTX 2080 Ti |
| cap250 sealed wall time | 2,554.9166 s (v5.1b 1,287.04 s); ratio 1.98 matches the 2080 Ti warm ratio 1.97 | verified receipt; worker is the unknown | `docs/STATE.md` L8, `friday/DECISION_18NOV.md` L64–68 |
| Sealed session count / windows | not recorded anywhere in the repo | **unknown** | grep of docs, friday, evidence |
| Package size limit | none in YAML or participate pages. Cap250 is 66.8 MB and uploaded fine. The script fails above 90 MB | **no stated limit (verified absent)** | grep of submodule YAML and pages |
| Codabench storage | 15 GB per profile shared across tracks (docs). On 12 Nov the upload was refused ("Insufficient space": 58.7 MB free, 62.3 MB needed); free space rose to 130 MB minutes later after two old submissions disappeared without our action. v5.2 (85 MB) needs ~85 MB free at upload time | **repo-claimed** (observed once); cause **unknown** | `docs/COMPETITION.md` L93, `docs/SEALED_UPLOADS.md` L409–413 |
| Final ranking rule | "the best of each team's **last five** sealed submissions" | **verified** (pinned submodule page) | `2026-competition/codabench/pages/timeline.md` L15; `competition_bci.yaml` L65 ("final NeurIPS ranking is taken from this phase") |
| Our window | 6 of 7 used (STATE). The last five would be #3–#7: 1048575, 1050185, 1058969 (cap250) and the final. Cap250 stays inside the window, so the last upload cannot lower our best. The YAML cap is 10, the live cap 7 | verified rule; live count **repo-claimed** | `friday/DECISION_18NOV.md` L24, L84 |
| Deadline | YAML end 2026-11-21 23:59:59 (zone unstated); the timeline page says "Anywhere on Earth" (= 22 Nov 12:00 UTC). The signed-in phase page showed Saturday 21 Nov 23:59 CET. Plan on **21 Nov 22:59 UTC (23:59 CET)** | YAML **verified**; CET reading **repo-claimed**; AoE reading **unverified** | `docs/SEALED_UPLOADS.md` L3, `friday/DECISION_18NOV.md` L24 |
| Quota | 1 per UTC day; a failed run uses no quota, a cancelled one does. Queue was large on 18 Nov; cap250 ran 42.6 min | repo-claimed | `docs/SEALED_UPLOADS.md` |

Consequences for Friday: upload by about 18:00–20:00 CET leaves a long buffer for the queue; the 21 Nov UTC slot remains the retry only if the Friday run fails (failed runs cost no quota). Re-read the live phase page for quota, free space and the end time before any upload.

## 2. The tool

```
python friday/qualify_package.py local   CAND OUT [--expect-sha H] [--expect-score S] [--skip-replay]
python friday/qualify_package.py cluster OUT [--ref-zip Z] [--submit]
python friday/qualify_package.py report  OUT [--fetch]
```

- **local** (about 4 min on the Mac). CAND is a flat package folder or a finished ZIP. A folder becomes a deterministic ZIP (sorted, fixed timestamps), a ZIP is copied byte for byte. Writes `OUT/package.zip`, `OUT/local.json`: SHA-256, size, 39-style member list with per-file hashes, CRC test, layout gates (root `submission.py`, no other `.py`, no nested entries, suffixes `.py .npz .pt .so`, ≤ 90 MB), official CPU replay through `scripts/replay_package.py` (official ingestion and scoring, files unchanged), and real-window checks through the official `fresh_model` path: reset (S1, then S2, then S1 again equal), prefix (labels of the first 30 equal on a 30-window run), future perturbation (noise after window 30 leaves labels 1–30 unchanged). Last line `LOCAL PASS/FAIL`.
- **cluster** writes `OUT/cluster/{a100,rtx}.sbatch`: inside the pinned worker SIF, replay ref, cand, ref, cand with `--nv` on `a100-galvani` and on `2080-galvani` (RTX 2080 Ti). `--submit` copies both ZIPs and submits over `ssh mlcloud-galvani` (eduVPN on). Hashes are checked on the cluster before any GPU time.
- **report** (`--fetch` copies results back) projects the sealed time as `2554.9 s × cand_wall / ref_wall` per device from the minimum of the two repeats, requires identical repeat scores and files unchanged, and prints `QUALIFY PASS`, `FAIL` or `PENDING-CLUSTER`.
  The ratio form is the only transferable number: the sealed session count and worker are unknown, and cap250 stock repeats on RTX have changed labels in rare cases (bugs_codex §4, item 2), so a candidate close to the budget needs the extra RTX long-stream timing the review asks for.
- Not covered, by design: replay/check.py (v3 chain only), shuffled-delivery accuracy, SHA pinning of `submission.py` code, and anything numerical across devices beyond the repeat-score equality.

## 3. Smoke on exact cap250 (19 Nov)

`python friday/qualify_package.py local runs/sealed/final_zips/v66_cp1_cap250.zip OUT --expect-sha 9c4d5538… --expect-score 0.9945788530`

- SHA-256 `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`, 66,788,282 B, 39 members: **matches the known hash**.
- Official CPU replay cell-mean BA **0.994578853046595** (known 0.9945788530, `docs/V6_BUILD.md` L529), files unchanged, ingestion 160.6 s: **matches**.
- Reset, prefix and future checks pass; `LOCAL PASS`. Bugs found and fixed while smoking: benchopt not on PATH, missing `BENCHOPT_DATA_HOME`.
- Cluster stage smoke (cap250 as both ref and cand): see the section below.

## 4. Friday timeline (≈ 1 h 15 min plus queue, budget 3 h)

1. Candidate folder or ZIP ready, ledger line written (T+0).
2. `local` (4 min) in parallel with `cluster --submit` (copy 2 × 67 MB, a few minutes; A100 4 replays ≈ 15 min, RTX 4 replays ≈ 30–40 min plus queue).
3. `report --fetch` gives the PASS line and projection; paste into the hand-over note with the SHA.
4. Hand over the ZIP; the user checks live quota, free space and phase, and authorises the upload. If free space is short, ask the user to remove old submissions first.

## 5. Cluster smoke result

Not finished when this note was written. The 67 MB upload to Galvani runs at about 6 MB/min (about 11 min per ZIP,
so budget it first on Friday; the reference cap250 ZIP is symlinked on the cluster, not re-uploaded). `cluster --submit`
was left running in the background on the Mac (it submits `a100.sbatch` and `rtx.sbatch` once the upload ends).
Finish with: `python friday/qualify_package.py report <OUT>=scratchpad/q2 --fetch` (needs eduVPN on). Expected for the
self-comparison: ratio about 1.00, projected sealed time about 2,555 s, `QUALIFY PASS`. Until that line is read, the
cluster stage of the tool is untested end to end (sbatch syntax checked with `bash -n`; report parsing checked only on the pending path).
