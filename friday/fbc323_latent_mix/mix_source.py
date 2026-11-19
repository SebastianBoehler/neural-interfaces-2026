"""Names and paths for the matched latent interpolation experiment."""
from pathlib import Path


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "within", "cross")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{k}_{m}" for k in KINDS for m in MODES)
