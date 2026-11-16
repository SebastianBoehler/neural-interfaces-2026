"""Run native-front parity checks without requiring pytest on Galvani."""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_covariance import test_covariance_blocks_real_window  # noqa: E402
from test_preprocess import test_real_window_car_bandpass_and_slow  # noqa: E402
from test_tangent import test_native_causal_tangent_reset_and_prefix  # noqa: E402

with tempfile.TemporaryDirectory() as directory:
    test_real_window_car_bandpass_and_slow(Path(directory))
    test_covariance_blocks_real_window(Path(directory))
test_native_causal_tangent_reset_and_prefix()
print('front-end components: real-window filters/covariances and tangent/reset parity passed; no score', flush=True)
