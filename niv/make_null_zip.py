"""Derive a placebo package: identical to SRC except the fine-tuning augmentation seed (v6_ft_seed 0 -> SEED).

Null calibration of NI-V: same configuration, different seed. Only weights.npz changes; submission.py is untouched.
python make_null_zip.py SRC.zip OUT.zip SEED
"""
import io
import sys
import zipfile

import numpy as np

src, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
    for info in zi.infolist():
        data = zi.read(info.filename)
        if info.filename == "weights.npz":
            w = np.load(io.BytesIO(data), allow_pickle=False)
            d = {k: w[k] for k in w.files}
            assert int(d["v6_ft_seed"]) == 0 and d["v6_ft_seed"].dtype == np.int64
            d["v6_ft_seed"] = np.array(seed, np.int64)
            buf = io.BytesIO()
            np.savez(buf, **d)
            data = buf.getvalue()
        zo.writestr(info, data)
print("wrote", out)
