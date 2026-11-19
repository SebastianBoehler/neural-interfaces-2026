# F-C4 Galvani qualification receipt

Smoke 3127413 ran on the NVIDIA A100 PCIe 40 GB. It started at 14:09:24 CET on 18 November after 5m40s in queue, then completed exit 0:0 in 4m50s. The pending smoke reservation was shortened from 22 to 10 minutes based on measured F-C3 smoke time. All source, stock, route and replay checks passed. No task labels were read.

Each arm ran five genuine refits per Scherer/G prefix, over the same two 61-window prefixes. The ten unequal-mass refits passed the per-event direction guard. Balanced reduced the dominant pseudo-class's mean share of normalized CE weights by 3.344 pp. Anti-balanced increased it by 3.400 pp. The balanced shifts by task index were −3.242 pp motor imagery, +1.877 pp subtraction and +1.364 pp word generation. These are changes in loss coefficients, with parameter gradients also determined by logits and waveforms. [Coefficient report](cap250-c4-mass-smoke.json)

Galvani counted 24 active and three running allocations in account gehler immediately before release. Array 3127453 was accepted with nine paired jobs, an A100 concurrency limit of three and a 22-minute limit per task. Collector 3127454 waits on all nine succeeding. Both were pending on first inspection. Source commit `87de899`, manifest `be10f8ec3cb19aeb9faa94616343ab994685ac28198e4f63920274c0fe37f65d`. Remote directory `/mnt/lustre/work/gehler/gfh527/ni26_friday_c4_v1_20261118`.

[Preregistration](cap250-c4.md)
