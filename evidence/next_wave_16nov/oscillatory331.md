# §331 independent raw oscillatory state-space pilot

Prepared17November2026,06:10CET; no real-data execution or accuracy yet.
This is additional mechanism preparation, not the scheduled06:51≥15-minute research block.
The scientific-discovery and engineering-workflow skills apply. Root retains project history; no blind/fresh-model claim.

Working assumption: useful task information may require input-dependent within-window dynamics.
Success check: independent recurrence/gradient parity and actual active learning, then matched S−F source contrasts.
Current4-second input is complete; no session history, later window, external data or pretrained weights are used.

The body has two32-wide oscillatory residual blocks over all480samples at120Hz.
It preserves EEG-band frequencies with learned bounded.5–55Hz angles and positive decay times.
Static and selective dynamics share initialization; local/no-recurrence is a separate capacity control.
The same287source-scaled mean/log-band-power branch keeps amplitude information available.
Three whole decoders plus unchanged raw SessionBN reference fitP04–P10; P02/P03are exposedqueries, P01excluded.
All source models/guards/banks precede query labels. No automatic outer/chain/package expansion.

Primary grounding:

- [S5](https://arxiv.org/html/2208.04933v3),9Aug2022/revised3Mar2023, motivates associative affine recurrence scans.
  Our straightforward log-depth scan does extra work and is not a claim to match the paper's efficient implementation.
- [S7](https://arxiv.org/html/2410.03464v1),4Oct2024, motivates stable input-dependent state transitions.
  Our contractive complex-diagonal parameterization differs; no S7 theorem or benchmark is claimed for it.
- [NeuroMambaNet](https://www.mdpi.com/2504-2289/10/11/385),11Nov2026, studies cross-person MI on different datasets.
  Its EEG-task result is motivation only; the repository license issue prevents importing its code here.
- [LuMamba](https://arxiv.org/abs/2603.19100),19Mar/revised4Jun2026, combines channel queries and bidirectional state-space EEG learning.
  Clinical tasks/pretraining differ. Its [official repository](https://github.com/pulp-bio/BioFoundation) licenses codeApache2,
  weightsCC-BY-ND4; no weights/code are used in this independent implementation, and fine-tuning permission is not assumed.

The novelty claim is only a previously unexecuted architecture/mechanism in this project.
Its scientific merit, useful EEG information, generalization and speed are unmeasured.
See prospectiveLEDGER§331 for exact predictions, fixed training, resource bounds and refutation rules.

Execution ownership update17Nov: preparedmodel.py remains unchanged. InitialCPUguard3117534(source3a37fb7e…4e627)failed because a randomtimingprobe passedunsortedindices to the existingselected-labelreader; correct readerrejection, beforeanypilot/querylabels. Preserved`oscillatory331_guard_v1_failed.txt`. Newimmutablev2sorts ONLYthose32engineeringprobeindices, leavingtrainingbatchschedule/recipe/tolerances unchanged. V2CPUguard3117539queued/running in`/mnt/lustre/work/gehler/gfh527/ni26_oscillatory331_v2_20261117`, source`b0fec93e6c3c6e3f66cc1d967e2c3b7ba1c87e01ca8308d3d56a6f5e0fe8fde9`. It recomputes independentcomplexoracle, gradient/state/batch checks andactualconservativecostgate before anysourcepilot.
