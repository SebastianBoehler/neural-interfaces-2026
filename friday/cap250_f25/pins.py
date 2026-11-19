"""Immutable F25 and F-C20 parents for the held-new corrected replay."""
from pathlib import Path


BASE = Path("/mnt/lustre/work/gehler/gfh527")
CAMP = BASE / "ni26_friday_causal_rebuild_20261118"
F25 = BASE / "ni26_friday_s1c_modulation_20261119"
C20 = CAMP / "fc20_outer_v1"
C20_PUPILS = CAMP / "fc20_pupils_v2"
ACTOR = "589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c"
F25_SCORER = "939a96514d432cedebf24a0c4500927f71401ed76117adc73fc76604453a779c"
F25_BLIND = "d281d5ecaa0f15a2757d4997da57b16188aff10e0d70f643074cd899d2140005"
F25_RESULTS = "64d2182825146f30486781e7a847407556a6acb32cdec55e08ecfc35182d52af"
C20_CONSUMER = "1319f8c515cba34eb9c7347c7b94a584ae2f33b3225c01ecfb623c9f60506f5b"
C20_PUPIL_SOURCE = "5925163a6331fc062a51f80f8262c05f7c12f069cfdba86d9e2cddf1f57d349c"
C20_SCORE = "1100016fe85b769282b01d7df6bb374e5590f1707ca8375e2aa1e84ba7b4ae30"
C20_RECEIPTS = dict(within0="a2249de5725ed36472ef6126ce1195b502fe1f3b9a47bef3124edf02bb81ebae",
                    full0="1e3109313d14fb2d1a429762b1f2ef544bed3bb38759e8b1bbd0d0f12f8ac683")
C20_ARRAYS = dict(within0="73f14c4082174008aa43ab9b7447a48a85973eb12a2113b8723520177de9f3a5",
                  full0="1a5f0c2d9cb81e38cd8f11608440c237c032ec413efee4ad9675a9f7507b9a57")
PUPIL_ARMS = ("hard", "single", "multi")
ARMS = ("stock", "identity", *PUPIL_ARMS)
ORDERS = ("full0", "within0")
