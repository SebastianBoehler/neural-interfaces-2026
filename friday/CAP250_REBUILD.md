# Rebuild the sealed cap250 payload

19 November 2026, 07:09 CET. The original v6.6 builder reconstructed **all 39 archive members byte-for-byte** from the archived fitted inputs. The rebuilt `weights.npz` matched exactly, and all **996 inherited arrays** and **31 inherited trace files** matched the sealed v5.1b payload. Both original ZIPs passed CRC and the frozen SHA-256 manifest. This verifies reproducible assembly from fitted inputs. Reproducing all source fits from the organizer’s supplied training split remains open.

| Stage | Authenticated input | Source and audit scope |
|---|---|---|
| Inherited models and fitted parameters | `runs/sealed/cand_v51b_own`, `weights.npz` SHA-256 `6a1f35c1282d98b9c2ca9bcb9b062acdb656f5c760c7a997b569102257481936` | The inherited set contains five six-seed SessionBN families and the external teacher trace. Source fit and export references are in [SessionBN](../../neural-interfaces-2026/docs/AUDIT_SBN.md) and [EEGNeX/IFNet](../../neural-interfaces-2026/docs/AUDIT_NXIF.md). |
| Eye and Rest | `runs/v6_members`, three traces of each family | [Build history](../../neural-interfaces-2026/docs/V6_BUILD.md) identifies the Eye nomix and Rest Shallow recipes. [Original training receipts](../../neural-interfaces-2026/evidence/v6_build_13nov/train_v6_members.jsonl) SHA-256 `f244ca3fe156d30da6aad9d1331f80e3484ddbfcedb306e3ebae0a98c64fad9d`. |
| CountFilter | `runs/countfilter/cf_params.npz`, SHA-256 `33b29e209f29e1c0ea0447e2ded56d7850346ae2a04611124d736f01a4ee9b62` | [Count calibration](../../neural-interfaces-2026/lean/tools/fit_cf.py). |
| TSEM prior | `runs/tpri/tpri.npz`, SHA-256 `07725495981b14e3b3e3bfe2eb65a6e9952f57b4cd41a4e856ecb3ddc8364207` | [Prior builder](../../neural-interfaces-2026/lean/tools/fit_tpri.py). |
| Frozen Solver | `submissions/lean_v66/submission.py`, SHA-256 `37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f` | [Committed v6 builder](../../neural-interfaces-2026/lean/tools/build6.py) SHA-256 `e048190516de9a56975899ea6b83a833631053dfc29b2a822fa84241dd779625`. Official evaluator revision `a66bc23d1a0e686b895c093a642b45c74970758b`. |

Run from the original repository with an unused `out` directory. The builder imports the assembled Solver as part of the check. Disable bytecode writing so its generated cache stays outside the payload inventory.

```bash
out=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/2026-competition" .venv/bin/python lean/tools/build6.py \
  runs/sealed/cand_v51b_own runs/v6_members "$out/cap250" \
  --v66 --t0 20 --every 10 --steps 5 --lr 0.001 --labels v51b --routed 1 --cap 250 \
  --session-s 1000000000 --tpri-on 1 --tpri runs/tpri/tpri.npz --tpri-kappa 20 \
  --cf-on 1 --cf runs/countfilter/cf_params.npz --relabel 1 --cp-on 1
```

The verified output file manifest has SHA-256 `250a0ca2ceb98a773d99c910fb7860387fb3c3d7cf78308378b8fff9e1420738`. The comparison receipt is `8dbcb4e9e8494175f58f7c9e9ceaeba68b7233f60f467b414f02bc90ac89529f`. Archive `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` remains the authority for submission 1058969. [Sealed receipt](DECISION_18NOV.md#official-penultimate-receipt) · [Official audit rule](https://neural-interfaces26.github.io/leaderboard.html#scoring)
