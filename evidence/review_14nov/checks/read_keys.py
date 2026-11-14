"""Read only the small config arrays of a package's weights.npz (zip members under 4 kB); no model weights are loaded."""
import io, sys, zipfile
import numpy as np

for pkg in sys.argv[1:]:
    with zipfile.ZipFile(f"{pkg}/weights.npz") as zf:
        infos = {i.filename[:-4]: i for i in zf.infolist()}
        print(f"== {pkg}: {len(infos)} arrays")
        for k, i in sorted(infos.items()):
            small = i.file_size < 4096
            show = k.startswith("v6_") or k in ("rb_alpha", "rb_k", "rb_w", "em_nmin", "em_iters", "em_bins", "em_ev_bins", "tsem_nmin", "route_t_start", "bio_t0",
                                                 "eb_T", "own_kref", "own_beta", "n_nets", "route_slot_g", "ts_k", "ts2_k", "ts_nsub", "ts2_nsub", "tsem_rho", "route_rho")
            if show and small:
                a = np.load(io.BytesIO(zf.read(i)), allow_pickle=False)
                print(f"  {k:22s} {a.dtype} {a.shape} {a.tolist() if a.size <= 12 else '...'}")
            elif show:
                print(f"  {k:22s} ({i.file_size} bytes, not read)")
        for k in ("own_bands", "ts_bands", "ts2_bands", "tsem_V", "route_V_hi", "own_prior", "em_ev_q", "member_weights"):
            if k in infos:
                with zf.open(infos[k]) as f:
                    v = np.lib.format.read_magic(f)
                    shape, _, dt = np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f)
                print(f"  {k:22s} header only: {dt} {shape}")
        # input-stream parameters of every net (header + scalar values only)
        keys = {}
        for n in range(200):
            if f"net{n}_align" not in infos:
                continue
            vals = tuple(np.load(io.BytesIO(zf.read(infos[f"net{n}_{k}"]))).tolist() for k in ("align", "k", "scale", "shrink"))
            f = str(np.load(io.BytesIO(zf.read(infos[f"net{n}_sjit_file"])))) if f"net{n}_sjit_file" in infos else ""
            keys.setdefault(vals, []).append((n, f))
        for vals, nets in keys.items():
            print(f"  input (align,k,scale,shrink)={vals}: nets {[n for n, _ in nets]}")
            print(f"     sjit: {[(n, f) for n, f in nets if f][:40]}")
