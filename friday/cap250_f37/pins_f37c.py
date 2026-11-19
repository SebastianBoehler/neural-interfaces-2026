"""Pin the independently sealed F37, F-C20 and F32c source and output banks."""
import json
from pathlib import Path

from locked32 import C20, C20_ARRAYS, C20_PUPILS, C20_PUPIL_SOURCE, C20_RECEIPTS


WORK = Path(__file__).resolve().parent
F37 = WORK.parent / "ni26_friday_s1c_spectral_shift_20261119"
F32C = C20.parent / "f32c_v2"
ACTOR = "0575236e532b5858a28c0e510d8fda3030cc5f4584648aa657642347e6f833bd"
SCORER = "70deb2a3c25e1a2a569afd435aea726e9722ad5e442440dfa00fb7e79a90e13d"
BLIND = "836eb58378fbd924a410af31f41ed364449652af3595bdfedbab810d7403113c"
RESULT = "1bffcf80e6726ae89edd54953676068a1f332215185334b1f9662d305e2b16aa"
F32_BLIND = "aaf7e95734d83cae19fb05d0305107138884a1600d7f2de3cf8ffb3b6e13ce6e"
F32_PUPIL = "4a7f749f7074665c046bac45e4d8eb70b88dd3dad7b68f2153ede473076706a6"
F32_SCORE = "58034507675e38823a563967ea2a19cb31d9a9ff64b30f1558923b42e7d3ac6a"
F32_RECEIPTS = dict(within0="148c94b68d27d51fb9bdc2eed7d181678a01781ad6fc529a14023b2cd9a92e35",
                    full0="6b8bc99731014ca663a72ce331cae7cbea12c6f4c9d2688e40da1f3e9743be49")
F32_ARRAYS = dict(within0="d76d3e24ab4ebba093b5721ac6c1fb70e91e9abce374ca149863cb5c64e00696",
                  full0="d690054a00d0e017ff1f7fc458e3568365b5bc7660df9e794f72662984327890")
DATA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
DELIVERY = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
DEVICE = "NVIDIA H100 80GB HBM3"
ORDERS = ("within0", "full0")
PUPILS = ("hard", "alpha", "beta")
ARMS = ("stock", "identity", *PUPILS)


def read_folds(sha):
    fixed = ((F37/"actor.sha256",ACTOR), (F37/"score.sha256",SCORER),
             (F37/"before_labels.sha256",BLIND), (F37/"whole_results.json",RESULT),
             (C20_PUPILS/"source.sha256",C20_PUPIL_SOURCE),
             (F32C/"pupil.sha256",F32_PUPIL), (F32C/"before_labels.json",F32_BLIND))
    if any(sha(p)!=digest for p,digest in fixed):
        raise ValueError("signed F37, F-C20 or F32c input manifest differs")
    source = {Path(p).name:h for h,p in (line.split() for line in (F37/"before_labels.sha256").read_text().splitlines())}
    comparators = []
    for root,digests in ((C20,C20_RECEIPTS),(F32C,F32_RECEIPTS)):
        folds = []
        for order in ORDERS:
            path = root/f"{order}.json"
            if sha(path)!=digests[order]:
                raise ValueError("signed H100 complete proof differs")
            m = json.loads(path.read_text())
            if m["held_labels_opened"] or m["order"]!=order:
                raise ValueError("H100 hard reference scope differs")
            folds.append(m["folds"])
        if len(folds[0])!=20 or folds[0]!=folds[1]:
            raise ValueError("reference H100 folds differ across deliveries")
        comparators.append(folds[0])
    if len(source)!=20 or any(set(source)!=set(ref) for ref in comparators):
        raise ValueError("F37 and reference fold coverage differs")
    return source,*comparators
