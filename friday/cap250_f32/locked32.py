"""Pinned F32, hard posterior and full-system anchors."""
from pathlib import Path

BASE = Path("/mnt/lustre/work/gehler/gfh527")
CAMP = BASE / "ni26_friday_causal_rebuild_20261118"
F32 = BASE / "ni26_friday_s1c_motor_joint_20261119"
C20 = CAMP / "fc20_outer_v1"
C20_PUPILS = CAMP / "fc20_pupils_v2"
ACTOR = "f8d074e6867ebfda82383c164c257d64f154b251b0ccdbcb5153b297e7917723"
F32_SCORER = "49458dc244ca5d0583ab09aa4b84bca50ecce59f1d34777fb46d80d9e57e9762"
F32_BLIND = "ad428671148719a6d2e6c2cda39aaa8a7ffe27d348ea143e2668dcc30eeaa959"
F32_RESULT = "1b9dd9d99b655d1dd5f1fff3655995d26593e6373bab6729fb64107b46daf5df"
C20_PUPIL_SOURCE = "5925163a6331fc062a51f80f8262c05f7c12f069cfdba86d9e2cddf1f57d349c"
C20_SCORE = "1100016fe85b769282b01d7df6bb374e5590f1707ca8375e2aa1e84ba7b4ae30"
C20_RECEIPTS = dict(within0="a2249de5725ed36472ef6126ce1195b502fe1f3b9a47bef3124edf02bb81ebae",
                    full0="1e3109313d14fb2d1a429762b1f2ef544bed3bb38759e8b1bbd0d0f12f8ac683")
C20_ARRAYS = dict(within0="73f14c4082174008aa43ab9b7447a48a85973eb12a2113b8723520177de9f3a5",
                  full0="1a5f0c2d9cb81e38cd8f11608440c237c032ec413efee4ad9675a9f7507b9a57")
PUPILS = ("hard", "detached", "joint")
ARMS = ("stock", "identity", *PUPILS)
ORDERS = ("within0", "full0")
