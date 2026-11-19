# Assumption sweep A · rules, constraints and limitations (Opus, 19 Nov 2026 evening)

Read-only audit. Sources: pinned submodule `a66bc23` (14 Oct), **upstream `main` at `8447e181` (19 Nov 13:10 UTC,
"align final sealed quotas, runtime and score publication")**, fetched with `gh api` into the scratchpad and not into
the submodule. Also used: neural-interfaces26.github.io `rules.html`, `tracks.html` and `participant-guide.html`
(fetched 19 Nov; `faq.html` now redirects to `rules.html`), the cap250 ZIP's `submission.py`
(= `submissions/lean_v66/submission.py`, 1,722 lines), our docs and `data/proteus_windows.npz`.
Sibling audits: (1) sealed composition, (2) data-path bugs, (3) error decomposition and design. Rows that belong to them
are only cross-referenced here.

Class: **OFF** = official rule (current primary text) · **ORG** = organiser statement outside the rules (Discord) ·
**INF** = our inference · **SELF** = self-imposed. "Verified" means checked against current primary text today.
**Bold rows**: self-imposed or mis-read, and plausibly worth ≥ 0.3 pp (or the same size in risk).

## 0. Headline: the pinned submodule is stale for rules

Our pinned copy is 8 commits behind upstream. The pin stays correct for code (`streaming.py`, `objective.py` and
`metrics.py` are unchanged). Its rule pages are stale. The newest upstream commit (`8447e181`) and `rules.html` now
state the following.
- Sealed workers use **one NVIDIA RTX PRO 4500** (Blackwell, sm_120, 32 GB). Tracks 02–04 have a **60-minute
  configured budget** (Track 01 has 120 minutes). "Evaluations that exceed their execution limit time out." The worker
  task timeout is 20 minutes longer (80 minutes), "This does not increase the configured execution budget."
  `competition_bci.yaml@8447e181:79` still reads `execution_time_limit: 3600`.
  → The Discord "limit was raised / 4,800 s" reading (`OFFICIAL_CONTRACT.md:71-77`, `friday/DECISION_18NOV.md:64`,
  `evidence/wave_19nov/laneF_package.md:10`, `strategy_opus.md` "headroom 1,000–2,200 s") is **superseded**. Budget
  against 3,600 s.
- Ranking: "Codabench automatically displays each participant's best eligible score … Only the **best eligible sealed
  score** from you or your team is retained." The leaderboard uses `submission_rule: Force_Best`
  (`competition_bci.yaml@8447e181:93`). The pinned rule "best of each team's **last five**"
  (`timeline.md@a66bc23:15`), cited as "verified" in `laneF_package.md:17`, is gone.
- Quota: 7 in total per **team** per track, shared across the members of the Codabench organisation, and at most
  1 per day. Queued or running submissions reserve a slot. **`Failed` does not count** toward either the daily or the
  total limit. Finished runs and runs cancelled by us count. Deleting a submission does not restore its slot.
- Deadline: **Sat 21 Nov 23:59 CET = 22:59 UTC**. The phase ends `2026-11-21 22:59:00` UTC. The pinned page's
  "Anywhere on Earth" is gone.
- The cohort text was weakened. The FAQ removed "ten participants absent from it" and now says "Exact cohort size …
  remain private". `competition_bci.md` dropped "20-participant, six-session". The 57 % new-people weight rests on
  withdrawn text. → audit (1).

## 1. Table

| # | Constraint (what we act under) | Exact statement · source | Class | Verified? | Where it binds in our code/docs | If relaxed: what changes, plausible pp |
|---|---|---|---|---|---|---|
| 1 | Runtime limit 3,600 s | "configured execution budget is … 60 minutes for Tracks 02–04 … Evaluations that exceed their execution limit time out" `terms.md@8447e181:42`; `rules.html` Rule 5; YAML `execution_time_limit: 3600` | OFF | **yes (19 Nov)** | `friday/qualify_package.py:25` `LIMIT_S=3600`; `OFFICIAL_CONTRACT.md:71-77` still says the limit was "raised" | Not relaxable. Correction in the other direction: drop the 4,800 s hope. cap250 used 2,555 s, so 1,045 s of headroom remains. Container overhead outside benchopt's `duration` (`ingestion.py:126-134` times only `benchopt run`) eats some of it |
| 2 | 80-min real timeout | "a timed-out attempt may remain `Running` until … 80 minutes in Tracks 02–04. This does not increase the configured execution budget" `terms.md@8447e181:44` | OFF | yes | `docs/STATE` (old), `map_allowed_space.md:182` | Not a budget. route3geb's 3,732 s finished under the 10 Nov rescoring, so enforcement has been lenient, but nothing guarantees it now |
| **3** | **Self-cap: projected time ≤ 0.85 × 3,600 = 3,060 s (Lane E: ≤ 3,300 s A100 central)** | our rule | **SELF** | n/a | `friday/qualify_package.py:26` `MARGIN=0.85`; `laneF_package.md:11`; `laneE_capacity.md:56-62` | **A Failed run (timeout included, if Codabench marks it Failed: unverified) uses no daily or total quota (`terms.md@8447e181:25`).** A runtime-heavier candidate uploaded **Friday** risks only time: if it fails, a safe package can go Saturday. The 15 % margin is not even the right size: worker speed swung by +1,045 s once (route3geb rescoring), while the RTX ratio proxy matched (1.97 proxy vs 1.98 sealed). Use the margin by day: Friday ≤ ~3,450 s projected, Saturday ≤ 3,060 s. Catch: a run still queued or running reserves the 7th slot, so the Friday upload must be early. Gain is real only if capacity arms convert time to accuracy (Lane E FT replicas, Eye/Rest 6+6: results pending). Their ceiling is the FT dose that took v5.1b→cap250 (+1.01 pp sealed for +1,268 s): plausibly **+0.1 to +0.4 pp** |
| **4** | **Sealed GPU "slower RTX, model unknown"; proxy RTX 2080 Ti** | "Sealed evaluations run on one NVIDIA RTX PRO 4500 GPU" `terms.md@8447e181:42` | OFF (was ORG/INF) | **yes (19 Nov)** | `laneF_package.md:12`; `friday/rtx/*`; the package keeps static FBCNet members as **float64 tensors on the GPU** (`lean_v66/submission.py:287-296`), disables TF32 (`:1715-1716`) and runs single-threaded (`:1708`) | Blackwell workstation GPUs have weak FP64, and TF32 is off for CPU/GPU parity only. Both are runtime spent on parity rather than accuracy; this is unmeasured on the real card (the 2080 Ti has weak FP64 too, which may be why its ratio matched). If FP32/TF32 static paths freed ~10–20 % of 2,555 s, that time funds row 3's capacity. Worth the same **+0.1 to +0.3 pp** as row 3, and only together with it. Do not change numerics on the final package without the same-input label check |
| 5 | Upload count: 7 per team, 1 per day; 6 used | `terms.md@8447e181:13-15,21-28`; `rules.html` | OFF | yes (6 used is from `DECISION_18NOV.md:84`) | `SEALED_UPLOADS.md`; per-upload user authorisation (AGENTS.md:39) | No extra slots: teammates share one quota (the pinned terms' "individual submission quota" is superseded), and registration is closed |
| **6** | **Ranking: best eligible sealed score (Force_Best). The last upload cannot lower our rank** | `terms.md@8447e181:30-34`; `timeline.md@8447e181:23` | OFF | **yes** | Selection rule `friday/OVERNIGHT_TOP3.md:150` ("favor stronger held-person and external evidence inside development noise"); "misses advancement" verdicts (`docs/STATE.md:28`, F-C25 hard; FBL2; F-C22); `laneF_package.md:17` cites "last five" | **With a protected floor, the last slot is a free call option.** The right criterion is E[max(0, S_cand − 0.82987)]. It rewards upside and variance, not a positive paired lower bound. Re-uploading stock has ≈0 value: the order is fixed and identical for all submissions, so only RTX nondeterminism differs. Gates written for "do no harm" (exact-archive external ≥ 0, RTX numeric tolerance 5e−5, paired lower bound > 0) protect a floor that cannot be lost. Ranking candidates by mean held-new + external gain with a runtime check alone would have kept F-C25 hard (+0.425/+0.374 held-new A100; pooled external +0.220 [+0.018,+0.432]; exact archive +0.050) and FBL2 (+0.383/+0.488) as live options, not "misses advancement". Stake: picking the best-mean variant versus stock: expected **+0.1 to +0.3 pp**, with an upside tail of ≥ +0.5 pp. Caveat: the A100 refit-trajectory gains may not survive RTX numerics (the cap250 vs v5.1b gap fell 0.73 pp on 2 people with the 2080 Ti) |
| 7 | Deadline | "Saturday, November 21 at 23:59 CET (UTC+01:00)" `timeline.md@8447e181:3,19`; phase end 22:59 UTC | OFF | yes | `laneF_package.md:19` (lists AoE as "unverified"; it is now dead) | None. Queue time is separate and was long on 18 Nov. With one slot left, upload Friday so that a Failed run can be retried |
| 8 | Never cancel a run | "Finished or cancelled by you: counts"; "leave queued or running submissions intact" `terms.md@8447e181:26,44` | OFF | yes | `SEALED_UPLOADS.md:298` | Already followed |
| 9 | One window in, one integer label out, immediately; output final | `streaming.py:123-130` (prediction appended per call); FAQ "One integer class index per window" | OFF | yes (code) | `Solver`/`predict` | No buffering, delayed emission or revision is possible. The mechanism forbids it, not only the rules |
| 10 | Causality: current and **previously delivered** windows only | "a model may use the current and previously delivered windows, but no future window or label" `scientific_faq_bci.md@8447e181:95-96,104-105` | OFF | yes | whole package; our re-labelling of earlier rows at age t (`submission.py:1138`, r15 relabel) | Allowed as is. Transductive work over the delivered prefix (re-fitting, re-labelling, clustering the prefix) is explicitly in scope. Nothing more is available |
| 11 | Delivery order: fixed, identical for all submissions, not chronological; scope (session/run/interface) unstated | "the same evaluation order is used for all submissions" FAQ@8447e181:111-116; `tracks.html` "Windows no longer follow acquisition chronology"; participant guide "same evaluation order for all submissions" | OFF + INF (scope) | yes | count filter urn/latch (`submission.py:1180-1255`); within0/full0/runs0 instruments | Shuffle scope → audit (1)/(3). A fixed order also means our sealed scores across uploads are paired: sealed differences between our packages are low-noise |
| 12 | Session boundary: fresh `deepcopy` + `reset_state()`, nothing crosses (module variables included) | `streaming.py:27-43,95-105`; FAQ "State, windows, statistics, and parameter updates must not cross" | OFF | yes | AGENTS.md:17; v6.2i removed the process clock (`submission.py:31-33`) | Correctly read. Load-time precomputation (in `load_model`) is shared legitimately by every copy, and so is process-level CUDA warm-up. Only state that changes later predictions is barred. No loss found |
| 13 | Graz and BrainHero share session state; metric splits them | FAQ "State may persist across Graz and BrainHero runs in the same session"; `objective.py:86-89` | OFF | yes | session-level stream; no interface detector (`OPTIMIZATION_R10.md:257-262`, closed at alignment scope) | A per-interface count prior needs interface inference from EEG (INF, untested at that scope). Public cells are nearly balanced anyway (row 15), so the ceiling is small: ≤ 0.2 pp. Not bold |
| 14 | No metadata to the model; identity inference from EEG | FAQ "Participant, session, run, interface, and label metadata are not passed"; guide "Do not rely on undocumented … subject identifiers" | OFF + ORG (identity inference allowed, Discord 27 Oct / 10 Nov, second-hand) | primary text silent | gates/RouteB/personal heads | Already exploited (returning routing). Declare it in the write-up. The risk sits in the audit, not the rank |
| 15 | Class balance per cell | Not stated officially. Public data: **every Graz cell is exactly 32/32/32**; BrainHero cells are 29–35 per class (90–103 windows); 54 of 89 runs are 10/10/10 and 17 are 12/12/12 (`data/proteus_windows.npz`, computed today) | INF (strong, public) | public only | rebalancing, count filter urn, D-c causal output balancing (`laneD_prefix.md:59-98,161-168`) | Exploited. The causal balanced-assignment family is allowed (it uses only our own outputs). D-c: +0.517 full0, −0.205 within0, −0.183 Scherer, so undecided → audit (3). Metric detail: a predicted class absent from a cell's targets counts as recall 0 (`metrics.py:56-61`). Harmless while all classes are present |
| 16 | Metric: per participant×session×interface balanced accuracy, equal-weight mean | `objective.py:86-100`; `metrics.py:45-89` | OFF | yes | all scorers | Correct |
| 17 | Window and tensor: (1,41,480) µV, 0.1–75 Hz, notch 50/60; no raw 500 Hz at sealed | FAQ "What reaches predict"; "a submission cannot request raw 500 Hz sealed data" | OFF | yes | AGENTS "preserve the model-facing tensor" | Training-side preprocessing is free. `data/proteus_windows_start*.npz` already use other offsets for training |
| 18 | Public data: all 14 public sessions may train the final model | FAQ "represented participants contribute only their already-public sessions" | OFF | yes | Eye/Rest trained "all ten people" (`laneE_capacity.md:18,75`); members → audit (2) | Correct, if every shipped member was refit on all ten people. Check in audit (2) |
| 19 | External data and licences | "Public datasets may be used when their original terms permit competition use … Declare every external corpus, and a compute estimate" `terms.md@8447e181:38` | OFF + ORG (Scherer cleared on Discord, 11 Nov) | yes; Scherer is second-hand | teacher.pt, Scherer validation, PhysioNet/Lee/Shin/Schirrmeister pretraining | Not limiting. Owe a declaration + compute estimate at close |
| 20 | Pretrained models; staged HF cache | `participate.md:436-445`; `tools/staged_hf_models.txt` (dinov2-giant, reve-base); Dockerfile bakes `REVE_POSITIONS_PATH=/compet/reve_positions` | OFF | yes (image recipe) | `laneC_member.md:77` says the REVE position bank "is not staged and must ship": **mis-read**, the image bakes it | Low impact: REVE/CBraMod fusion chose weight 0 (`laneC_member.md:70-73`). A LoRA/adapter delta on staged `reve-base` would avoid the 415 MB issue if that line is reopened |
| 21 | Package size ≤ 90 MB | No per-ZIP limit. 15 GB of profile storage is shared ("practical limit is the space remaining") `terms.md@8447e181:46` | SELF (derived from the 12 Nov "Insufficient space: 58.7 MB free") | yes | `friday/qualify_package.py:27,64-65` | The user can free space by deleting archived non-leaderboard submissions. No measured-positive candidate needs > 90 MB, so ≈0 pp. Re-check live free space before upload |
| 22 | Allowed imports: worker image only; no compilers; torch cu130, braindecode, mne, moabb, sklearn, scipy, pandas, neuroai 0.4.0 | `participate.md:26-31`; `tools/Dockerfile` (`python:3.12-slim`, torch from the cu130 index); `requirements.txt` | OFF | yes (recipe; the live image tag was "v4-sealed" on 5 Nov, contents not inspected) | memory note "no compilers or numba"; C++ via bundled wheels allowed (ORG, 20 Oct) | `torch.compile`/Triton need a C compiler (slim image has none). CUDA graphs and TorchScript work. cap250 ships no `.so`. Not limiting accuracy |
| 23 | CPU count, RAM, GPU memory | Not stated. GPU memory 32 GB (RTX PRO 4500 spec) | INF | no | `torch.set_num_threads(1)` and `threadpool_limits(1)` (`submission.py:1708-1712`), measured 4× slower interleaved on a loaded A100 node | Unknown core count. Single-threaded CPU stages (eigh, EM, tangent) may waste the worker's cores. Speed-only, feeds row 3. Unverified |
| 24 | In-predict training allowed | "Label-free test-time adaptation is allowed only as specified in your track's Scientific FAQ" `participate.md@8447e181:40-43`. The pinned text "do not train" during `load_model`/`predict` was **removed** | OFF | yes | FineTune (`submission.py:1028-1171`) | Correctly used. Refit cadence, replicas and cap are self-chosen and limited by rows 1/3. `v6_ft_cap=250` never binds on public sessions (≤ 199 windows), so its effect is sealed-only (session length → audit 1). `v6_ft_session_s=1e9` is inert |
| 25 | Reproducibility audit (top 3 only), ±2σ, rerun of the training pipeline; σ undefined | `terms.md@8447e181:52-58`; audit 22–30 Nov | OFF | yes | "each arm must have a rebuild recipe before upload" (`SEALED_UPLOADS.md:20`); `CAP250_REBUILD.md` | Prizes only. Failing the audit "remain[s] on the public board". The rebuild recipe is SELF as an upload gate. Relaxing it changes no score, only prize eligibility. Keep it, but it should not veto a better-mean last upload (row 6) |
| 26 | Write-up obligations | method description, training + inference code, pre-training disclosures, compute estimate; ready "when the sealed phase closes" `timeline.md@8447e181:42`, Rule 8 | OFF | yes | none prepared in `friday/` | Not a score item. Must exist by 21 Nov 23:59 CET if we reach the top 3 |
| 27 | Cross-track award | "Best all-round performance · $2,000 … across multiple tracks" `prizes.html` | OFF | yes | not pursued | Outside Track 2 |
| 28 | Sealed logs hidden; status + aggregate score visible | `terms.md@8447e181:64-66` | OFF | yes | worker-image replays (`qualify_package.py`) | Correct |
| 29 | Composition (new vs returning share, session count) | "Exact cohort size, identities, session allocation, and evaluation order remain private" FAQ@8447e181:40 | OFF | yes | AGENTS.md:24 (57 % new; 0.2 P04 + 0.8 P10); Lane E "106 sessions" | → audit (1). The 57 % figure now rests on withdrawn text |
| 30 | Parity-first engineering (bitwise same-input checks, 5e−5 RTX tolerance as an advancement gate) | ours | SELF | n/a | `friday/cap250_hard_package`, F-C21g, `OVERNIGHT_TOP3.md:146` | Keep it as a bug guard (no-op detection). As a gate on the last upload it repeats row 6's error. Two untouched stock runs already exceed 5e−5 on the RTX. Covered by `strategy_opus.md:95` |
| 31 | "Score a part only inside the full system", ±0.3/±0.5 noise bars, preregistration | AGENTS.md:7-25 | SELF | n/a | every ledger card | → audit (3); `strategy_claude.md:26-48` |
| 32 | Local jobs: one capped MPS job; no work on login nodes | AGENTS.md:43-47 | SELF (safety) | n/a | `scripts/run_capped.sh` | Throughput only; HPC covers the heavy work |
| 33 | Never edit `submissions/*/submission.py`; keep the submodule pinned | AGENTS.md:31-34 | SELF | n/a | — | Correct for code. **For rules, read upstream `main`**: the pin hid four rule changes (rows 1, 4, 6, 7) |

## 2. Findings that plausibly cost ≥ 0.3 pp (or equal risk)

1. **Last-upload selection uses "do-no-harm" gates that protect a floor we cannot lose (row 6).** Force_Best, a fixed
   order and Failed-is-free make the 7th slot a free option. Choose the highest expected max(0, gain) candidate whose
   runtime passes. Do not re-upload stock. Do not require positive lower bounds or a non-negative external result.
   Stake: about +0.1 to +0.3 pp expected, with ≥ +0.5 pp upside.
2. **Runtime margin is set as if a timeout cost a slot (rows 3, 4).** A Failed run costs no quota. Upload a heavier
   variant Friday early and keep the 3,060 s rule for a Saturday fallback. It pays only if the capacity arms (Lane E)
   or FP32/TF32 savings on the now-known RTX PRO 4500 convert time to accuracy: +0.1 to +0.4 pp, unmeasured.
   Unverified: that Codabench marks a timeout "Failed" rather than "Finished" or cancelled. Ask on Discord before
   relying on it.
3. **Risk in the other direction: the "raised limit / 4,800 s" belief is dead (row 1).** Any plan budgeting past
   3,600 s (`strategy_opus.md`: "headroom about 1,000–2,200 s") must be cut back to ≤ 1,045 s over cap250.
4. **Stale pinned rule pages (§0).** `OFFICIAL_CONTRACT.md:71-77` and `laneF_package.md:10-19` should cite
   `8447e181`. No code change is implied.

Not found: any mis-read of causality, session reset, metadata or adaptation rules that blocks a legal method. The
contract is read correctly and tightly. Transductive use of the delivered prefix is allowed and already used. Look-ahead
and revision are impossible by mechanism. Class balance is near-exact on public cells and is already exploited
(undecided, audit 3).

Limits of this audit: no live Codabench page (bot-blocked, and logged-in views need the user). The Discord statements
are second-hand. The worker image contents and the RTX PRO 4500 FP64/TF32 runtime effect are not measured. Whether a
timeout counts as "Failed" is not confirmed.
