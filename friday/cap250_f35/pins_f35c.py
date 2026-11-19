"""Immutable F35 fit and same-device F-C22h reference releases."""
from pathlib import Path


WORK = Path(__file__).resolve().parent
F35 = WORK.parent / "ni26_friday_s1c_balance_a100_20261119"
REFERENCE = WORK.parent / "ni26_friday_s1c_modulation_a100_whole_20261119"
ACTOR = "b153df05e923c0b40795adaed413d8d024087e49e4b2be51445d81b6180db42e"
SCORER = "eaa05cb8ee29a01106c56cf5c17fd457ce607ab7426468f7e1e2046a7c7be6a4"
BLIND = "1ebeb5737c35c1a7e83083bfd4979db87e5bd22c37fb96d87d8a3c90bf84c530"
REF_BLIND = "439b4569aac56610992afea4e3ce87e5055402fc7d56477f14f395ecb0fa9657"
RESULT = "d22fcd1b314ac37cc3e6ee42b53b580e124209c08d77550db5df6ea91790a5ee"
REF_RESULTS = "01cef60a3838299076d23d761e67cd2e00724c4996062ed6e1fb81a761805dae"
REF_BANKS = dict(full0="845b830a4501eb2dff6868cfc2b94f558c4318f276ba28c7b7301c4df8174b4e",
                 within0="47c36e471d8a62aa96c1b343294422e274567a4a6d5318347be5600ada59b4b9")
DATA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
DELIVERY = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
DEVICE = "NVIDIA A100-PCIE-40GB"
ORDERS = ("full0","within0")
PUPIL_ARMS = ("hard","balance","mirrored")
