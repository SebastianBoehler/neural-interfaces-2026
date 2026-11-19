"""Official loader for an immutable offline fit."""
from pathlib import Path

from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "Team A clean-slate"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        folder = Path(meta["submission_dir"])
        package = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
        names = tuple(str(x).removeprefix("EEG ").split("-", 1)[0] for x in meta["ch_names"])
        if names != package["names"] or (meta["n_chans"], meta["n_times"], meta["sfreq"]) != (41, 480, 120):
            raise ValueError("Package montage and official EEG tensor disagree")
        dev = torch.device(meta["device"])
        recipe = package["recipe"]
        decoder = SpectralDecoder if recipe == "A2" else Decoder
        net = decoder(package["arch"], names, package["model"].get("positions"))
        net.load_state_dict(package["model"], strict=True)
        net.eval().requires_grad_(False).to(dev)
        for name in ("covariance", "gallery", "identity_scale", "enrolled"):
            package["bank"][name] = package["bank"][name].numpy()
        bank = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in package["bank"].items()}
        return SessionDecoder(net, bank, dev, recipe)
