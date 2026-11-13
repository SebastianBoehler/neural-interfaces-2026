"""Entry point for the competition harness."""
import numpy as np
import torch

from benchmark_utils.base_solver import CompetSolver

from .ensemble import Ensemble
from .inputs import channel_map


def check_package(z):
    """The code implements one configuration; a weights file that asks for another must fail here, not run as
    something else."""
    n = int(z["n_nets"])
    checks = {
        "member_weights = core1, n_nets nets, Hilbert core": len(z["member_weights"]) == n + 2,
        "both cores: common-average reference and sub-window covariances":
            all(int(z[f"{p}car"]) == 1 and int(z[f"{p}nsub"]) > 0 for p in ("ts_", "ts2_")),
        "core1 real, second core Hilbert": "ts_hilbert" not in z and int(z["ts2_hilbert"]) == 1,
        "rebalancing on": float(z["rb_alpha"]) != 0 and int(z["rb_w"]) > 0,
        "TSEM on": float(z["tsem_lam"]) > 0,
        "routing v3 with identity gate and own-space stage":
            int(z["route_v3"]) == 1 and "bio_V" in z and "own_beta" in z,
        "view = TSEM PCs + routing PCs": z["tsem_V"].shape[1] + z["route_V_hi"].shape[1] == z["route_gal"].shape[2],
        "one gallery row, head and own-space model per public person":
            len(z["route_gal"]) == len(z["route_ps1_W"]) == len(z["bio_gal"]) == len(z["own_prior"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"weights.npz does not match this code: {failed}")


class Solver(CompetSolver):

    name = "TangentShallowEnsemble"

    def load_model(self, meta):
        # per-window work is tiny: BLAS, OpenMP and torch run single-threaded (interleaved multi-threaded numpy and
        # torch calls were 4x slower on a loaded A100 node)
        torch.set_num_threads(1)
        try:  # threadpoolctl comes with scikit-learn
            from threadpoolctl import threadpool_limits
            self._limits = threadpool_limits(limits=1)
        except ImportError:
            pass
        # no TF32, so GPU and CPU agree to float32 precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with np.load(meta["submission_dir"] / "weights.npz") as f:
            z = {k: f[k] for k in f.files}
        check_package(z)
        model = Ensemble(z, str(meta.get("device", "cpu")), meta["submission_dir"])
        model.perm = channel_map(meta.get("ch_names"))
        return model
