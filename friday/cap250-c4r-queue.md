# F-C4r · Exact balanced-weight order replication

Registered at `8584b8e` after the original +0.119 pp result and before alternate-order replay. This measurement advances past the original interval-crossing condition to test order sensitivity on the same nine participants. The two production arms are exact staged stock and frozen F-C4 balanced weights. [Original comparison and optimizer coefficient checks](cap250-c4-result.md)

Source `fdb0d19`, A100 working directory `ni26_friday_c4r_v1_20261118`, manifest `6ba982b777e7dc9e0c33ab04bee3e01081129eb540f6f197d45619d9c51a108c`. The worker imports the original optimizer hook from authenticated F-C4 code and the original stream function from authenticated F-C3 code. Only the delivery schedule and experiment harness change. The old and balanced archives used for comparison are recorded in the original report. The tested package files remain read-only.

Real smoke 3128879 finished at 18:22:08 CET. It reproduced the 120-window Scherer/G exact stock trajectory and both arms' activity, the pre-refit labels, and fresh 61-window prefixes through five neural refits. The balanced arm showed the original dominant-class coefficient reduction after the float32 cast. Stock and balanced smoke SHA-256 digests were `3020ebc6ff0d607b75ac2217ccf17669f386edd4e56718f2d6cadbbc8e63b32f` and `94a8f91d64751999da14eead9d21c2f6ff23bc80f2b6a86728c4e17670731e49`. Recording-order replay printed `OK`.

Galvani array 3128892 runs nine independently paired A100 tasks at concurrency three. Each task has four sessions under orders 411 and 1411 and two independent clean worker processes. CPU scorer 3128893 authenticates all 18 decision files before scoring. The result measures delivery-order replication on previously examined Scherer people.
