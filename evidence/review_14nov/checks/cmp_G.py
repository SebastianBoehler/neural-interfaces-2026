"""Is the rest member's input stream the same computation as the nets' whitening stream? Compare the G arrays (13 kB) and scalars."""
import io, sys, zipfile
import numpy as np
pkg = sys.argv[1]
with zipfile.ZipFile(f"{pkg}/weights.npz") as zf:
    g = lambda k: np.load(io.BytesIO(zf.read(k + ".npy")), allow_pickle=False)
    a, b = g("v6_rest_G"), g("net0_G")
    print("rest G", a.dtype, a.shape, "net0 G", b.dtype, b.shape, "equal bytes:", a.dtype == b.dtype and a.tobytes() == b.tobytes(), "max abs diff", float(np.abs(a.astype(float) - b.astype(float)).max()))
    for k in ("v6_ft_budget_s", "v6_ft_cap", "v6_ft_labels", "v6_routed", "v6_ft_on", "v6_ft_t0", "v6_ft_lr", "v6_ft_every", "v6_ft_steps"):
        print(" ", k, g(k).tolist())
