"""Pinned independent archives for the exact held-new runs0 replay."""

from bridge_owner import MEANS, ROOT
from complete_zero import FEAT, STAGES

ORDER = "runs0"
ROUTE = ROOT / "runs/s281_v66_route/runs0.json"
ROUTE_SHA = "52e6797dd3bed19386779bef980b1efb206893c056b666aa5227628cb7963611"
FEATURE = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
BRIDGE_OLD = MEANS / "orders_v2/runs0/status.json"
BRIDGE_OLD_SHA = "edad644cbe09073285e7ee814fa4054ddd2cb7cf42ffa360a0051b529f375bbe"
PINS = {
    ROOT / "runs/s281_v66_cp/control_runs0_B2_pool.npz": "d11dfa8288c7484891240440bcdd95c212f3546dc8eb7d152673bcaee542ef23",
    ROOT / "runs/s281_v66_cp/control_runs0_B2_pool.json": "54083aa6b631f9b375fa1b10a23e85138978725c4e133d61956010b1076b5cf4",
    ROOT / "runs/s281_v66_cp/zero318_runs0_B2_pool.npz": "f65fe82ea2b3b927b3bcc4cf0c545b80699b9f4e0801a4f10c0e7f54b0ea7747",
    STAGES / "replay.txt": "5faba4e1c72f60850a7e557fea817212b48db99497111cd263639d29bbed3bdf",
    STAGES / "runs0.npz": "feda4b51981ca56690ac9baf3cab3a6e7f353838de927d618d0fd610c3586cbb",
    STAGES / "stages_runs0.npz": "f9e0db55cd578f79f684432d6ccab877c478068b19da367209ec3174938ebe23",
    STAGES / "stages_runs0.json": "e9e84b79e48f7506a632c222616d3603e59b6d20efe2f060097305e5de42f6ea",
    ROUTE: ROUTE_SHA,
    FEAT: FEATURE,
}
