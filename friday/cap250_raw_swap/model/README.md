# F-C13 one-window raw S1C actors

The two actors restore the frozen F16e hard-CE checkpoints trained with zero fixed-feature inputs. They use the saved source scale of 29.709604263305664 and the original FBCNet convolution and head. The FFT filter taps and constrained spatial weights come from each checkpoint. The head receives a constant zero vector in its 287 fixed-feature columns. There is no feature computation for those columns at runtime.

The bounded SessionBN uses the current window at weight one. It computes the window's bounded weight only for future windows. Its seven buffers store the count, unweighted and weighted moment sums, weighted mass and per-channel clipping state. The state belongs to one actor instance and resets for each session. There is no train-time or external label use.

## Load on the worker

The Galvani release is `/mnt/lustre/work/gehler/gfh527/ni26_friday_cap250_raw_model_v3_20261118`. Read `artifact.sha256` from that directory. The class order is motor imagery, mental subtraction, word generation.

```python
template = torch.jit.load(path, map_location=device).eval()
with torch.no_grad(), torch.jit.optimized_execution(False):
    template.reset()
session_actor = copy.deepcopy(template)
with torch.no_grad(), torch.jit.optimized_execution(False):
    posterior = session_actor(current_window)
```

The input is one contiguous float32 tensor of shape `[1, 41, 480]` on the same device. The result is float64 of shape `[1, 3]`. `reset()` returns a zero integer tensor. Keep the JIT context scoped to this actor. The thin `RawS1C` wrapper is available for callers that need a NumPy vector. Both artifacts loaded and reset on a real Scherer window in a clean process without importing Braindecode or Torchaudio. All graph operators are ATen operators.

| Actor or receipt | SHA-256 |
|---|---|
| Seed 0 F16e source checkpoint | `bcedeb0fb01b1735a834f9c8a7270d04f97fe085abce31a5c0bb05d21aa275f5` |
| Seed 1 F16e source checkpoint | `82b9ae356bc5e09131f3b5db3ced8674f1691e1839031c78e020844df233781d` |
| `raw_s1c_s0.pt` | `0dc677091668ad8acc6716a9bd75df6371829aa13ac6cd90985a3ef675126cdd` |
| `raw_s1c_s1.pt` | `cb16ec5919c8fc58d7a9a9f4c8defcb425a16b4d36359687a0a68a1b6b078634` |
| `model-receipt.json` | `2f8154a61736616dfc90caf1e55cbbb9a8fa25eb206fa9c1808161fc53a17508` |
| `artifact.sha256` | `2bcf07e5e2780b0e76e855a853cf1fbe5bdd8140356be868b746b25fb538d034` |
| `source.sha256` | `918ef52a559d8bf2c7a4b7c1d2816329ef744de7b8621b10cde88b487965bf11` |
| `ni26_worker.sif`, Torch 2.14.1+cu130 | `7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0` |

## Real worker smoke

Galvani A100 job 3131673 passed at 22:08:25 CET on 18 November 2026. Each seed and device replayed 480 raw Scherer G frames across both visits and full deliveries 411 and 1411. Every decision agreed with the resumed independent F16e batch oracle, which reproduced the locked historical decisions. Fresh copies agreed exactly on the first frame. The 31-frame truncated and interleaved-reset prefixes were exact. The frozen parameters and buffers remained bit-identical.

| Same fixture | Seed 0 | Seed 1 |
|---|---:|---:|
| Maximum absolute CPU posterior error | 4.03e−7 | 5.07e−7 |
| Maximum absolute A100 posterior error | 1.42e−5 | 9.92e−6 |
| Warm CPU forward, median ms/window | 5.63 | 5.65 |
| Warm A100 forward with host-returned posterior, median ms/window | 2.47 | 2.49 |

The first export, job 3131651, copied a CPU device constant into the FIR graph. The second export, job 3131665, corrected FIR portability. Its CUDA profiling executor changed the same first-window posterior by 3.52e−6 across fresh copies. The accepted actor uses the original FIR equations with movable taps and scopes JIT execution without profiling to the raw actor. The original state, scale, output schema, CPU/A100 posterior tolerance of 2e−5, and exact-decision and restart checks all apply. [F16e independent source and scorer receipts](../../fbc323_distill_external/result-18nov.md)

## Cohort runtime diagnostic

The same immutable A100 artifacts reproduced all **8,520** archived F16e decisions across nine Scherer people, two seeds, two visits and both pinned orders. The strict CPU-oracle posterior tolerance of 2e−5 passed for A, C, E, G and J. It was exceeded at least once for D, F, H and L. Across the whole A100 cohort, maximum posterior error versus the CPU batch oracle was 2.711e−5. The complete strict CPU-stream runs on five participants covered 4,740 decisions and a maximum posterior error of 5.070e−7.

Galvani 3131675 ran the eight additional strict checks. Independent diagnostics 3131687 and 3131688 completed the four cases that exceeded the strict tolerance. On those four people, the original eager A100 batch decoder differed from its CPU batch oracle by at most 4.807e−5. The A100 actor and eager A100 batch agreed on every decision and differed in posterior by at most 4.437e−5. The A100 actor's 31-window prefixes and resets remained exact. The run preserves the strict posterior failures and establishes decision parity for the frozen raw actor. This receipt measures the actor only. Gold labels were unopened.

The combined receipt is `/mnt/lustre/work/gehler/gfh527/ni26_friday_cap250_raw_model_diag_20261118/cohort-receipt.json`, SHA-256 `c0dfc84c694b1cb8490c6c1353b3617e15bf7c168ab91b9c06f19e91e3612d53`.
