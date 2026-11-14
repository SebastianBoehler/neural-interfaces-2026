"""Entry point for the competition harness."""
import numpy as np
import torch

from benchmark_utils.base_solver import CompetSolver

from .v6chain import Ensemble6
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
    checks.update({  # v6: the second chain for unrecognised sessions
        "v6: eye nets, rest nets, rest weight, fine-tuning schedule present": all(k in z for k in (
            "v6_eye_files", "v6_rest_files", "v6_rest_w", "v6_slots", "v6_ft_member", "v6_ft_t0", "v6_ft_every",
            "v6_ft_steps", "v6_ft_lr", "v6_ft_lam", "v6_ft_seed", "v6_ft_on", "v6_ft_labels", "v6_routed", "v6_ft_cap", "v6_ft_session_s")),
        "v6: one tuned member per family slot": "v6_slots" in z and len(z["v6_slots"]) == len(z.get("v6_ft_member", ())),
    })
    checks["v6.2c: count-filter switch v6_cf_on in {0, 1}"] = "v6_cf_on" in z and int(z["v6_cf_on"]) in (0, 1)
    if "v6_cf_on" in z and int(z["v6_cf_on"]) == 1:
        cf = ("v6_cf_W", "v6_cf_sw", "v6_cf_bands", "v6_cf_fs", "v6_cf_checks", "v6_cf_thr", "v6_cf_Ls", "v6_cf_lp", "v6_cf_cap",
              "v6_cf_tau", "v6_cf_floor", "v6_cf_K")
        checks["v6.2c: count-filter parameters present and shaped"] = all(k in z for k in cf) and (
            z["v6_cf_W"].shape == (41 * len(z["v6_cf_bands"]), len(z["v6_cf_sw"])) and (z["v6_cf_sw"] > 0).all()
            and len(z["v6_cf_thr"]) == len(z["v6_cf_checks"]) and z["v6_cf_cap"].shape == (len(z["v6_cf_Ls"]), z["v6_cf_Ls"].shape[1] + 2)
            and len(z["v6_cf_lp"]) == len(z["v6_cf_Ls"]) and float(z["v6_cf_tau"]) > 0 and float(z["v6_cf_floor"]) > 0
            and int(z["v6_cf_K"]) > int(z["v6_cf_cap"].max()))
    if all(k in z for k in ("v6_routed", "v6_ft_labels", "v6_ft_on", "v6_ft_cap", "v6_ft_session_s", "v6_ft_t0", "v6_ft_every",
                            "v6_ft_steps", "v6_ft_lr", "v6_rest_w")):
        checks.update({  # v6.2i: every switch and guard in its range (0 must not silently switch a guard off)
            "v6: v6_routed in {0, 1, 2}": int(z["v6_routed"]) in (0, 1, 2),
            "v6: v6_ft_labels and v6_ft_on in {0, 1}": int(z["v6_ft_labels"]) in (0, 1) and int(z["v6_ft_on"]) in (0, 1),
            "v6: guards v6_ft_cap >= 1 and v6_ft_session_s > 0": int(z["v6_ft_cap"]) >= 1 and float(z["v6_ft_session_s"]) > 0,
            "v6: schedule t0 >= 1, every >= 1, steps >= 1, 0 < lr < 1": int(z["v6_ft_t0"]) >= 1 and int(z["v6_ft_every"]) >= 1
                and int(z["v6_ft_steps"]) >= 1 and 0 < float(z["v6_ft_lr"]) < 1,
            "v6: 0 < v6_rest_w < 1": 0 < float(z["v6_rest_w"]) < 1,
        })
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
        model = Ensemble6(z, str(meta.get("device", "cpu")), meta["submission_dir"])
        model.perm = channel_map(meta.get("ch_names"))
        return model
