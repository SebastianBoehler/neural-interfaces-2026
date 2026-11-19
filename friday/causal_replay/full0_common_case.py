"""Pinned original full0 zero and matched common-FBLight/Sinc inputs."""

import json

from bridge_owner import BASE, MEANS, ROOT, pin

ORDER = "full0"
STAGES = BASE / "ni26_match456_stages_20261118"
FBL = MEANS / "full0_common_fbl_v1"
FBL_SOURCE = "71495c0952cc725855f1fec9461772b880a502653fb3ff097775a3f7d378144c"
ZERO = MEANS / "terminal_full0_v2/zero.json"
ZERO_SHA = "9b9dc4070b234a6df2c2d806c7ad5f2ddf2dbd231370fae688369a78b046663c"
ZERO_SOURCE = "fdc3f46b965354ad9180c63ff132c3ee680c7a3521bd272d3c692db9a027b2ac"
ROUTE = ROOT / "runs/s281_v66_route/full0.json"
ROUTE_SHA = "befe52d9c2491c15fbc613eb62ed45186969de69ee5b1cffe63906d834a600d7"
BRIDGE = MEANS / "full0_common_bridge_v1/status.json"
PINS = {
    ROOT / "runs/s281_v66_cp/control_full0_B2_pool.npz": "087b14f54b65496de07e728549d71501d0f4fdfd711794e10b41bec5f044fa7f",
    ROOT / "runs/s281_v66_cp/zero318_full0_B2_pool.npz": "d0a8fddef6e6f50c0b232c9760ba116e04a3a12ea60024964caafce07c83d4b0",
    STAGES / "replay.txt": "5faba4e1c72f60850a7e557fea817212b48db99497111cd263639d29bbed3bdf",
    STAGES / "stages_full0.npz": "09a21fb45aa7d7125a59137f792f55914ac65080af534f889360489a373aac91",
    STAGES / "full0.npz": "707d8a842fd4e924ffcb7f50ac5381e6a6c33abcd361abb30c89c185c8696f5a",
    ROUTE: ROUTE_SHA,
    ROOT / "data/proteus_windows.npz": "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327",
}


def qualified_zero(digest, order, path, source):
    if (digest, order, path, source) != (ZERO_SHA, ORDER, ZERO, ZERO_SOURCE):
        raise ValueError("the exact archived full0 control read differs")
    pin(path, digest)
    pin(path.parent / "source.sha256", source)
    proof = json.loads(path.read_text())
    if (proof["source_sha256"] != source or proof["order"] != order or not proof["exact_archive_zero"]
            or not proof["three_aux_exact"] or proof["held_labels_opened"] or proof["scored"]
            or proof["refit_matrix_opened"] or proof["stages"]["exact_interleaved_prefix_checks"] != 3
            or proof["stages"]["filter_affected_rows"] <= 0 or proof["stages"]["personalization_affected_rows"] <= 0):
        raise ValueError("the immutable full0 complete control differs")
