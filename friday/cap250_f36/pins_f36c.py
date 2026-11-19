"""Immutable F36 fit and same-device F-C22h reference releases."""
from pathlib import Path


WORK = Path(__file__).resolve().parent
F36 = WORK.parent / "ni26_friday_s1c_crossfade_a100_20261119"
REFERENCE = WORK.parent / "ni26_friday_s1c_modulation_a100_whole_20261119"
ACTOR = "baf7735eb9ca7f225b99ecbf58e6a16a8fea3a13813ae7c2e8434d0fff901510"
SCORER = "2734eeee0e398a53ffc5900b303255a2ef193df31d2adfb448b59eb1aba67367"
BLIND = "32bbcd0f1230c10ab88742ee6467a42b6bdef3788051644361d095093ed160f9"
REF_BLIND = "439b4569aac56610992afea4e3ce87e5055402fc7d56477f14f395ecb0fa9657"
RESULT = "86b65abf6273b296f5ef4fd8e4ce4712c8a1e7794aa16a3fb4c2bd31f6a7d08e"
REF_RESULTS = "01cef60a3838299076d23d761e67cd2e00724c4996062ed6e1fb81a761805dae"
REF_BANKS = dict(full0="845b830a4501eb2dff6868cfc2b94f558c4318f276ba28c7b7301c4df8174b4e",
                 within0="47c36e471d8a62aa96c1b343294422e274567a4a6d5318347be5600ada59b4b9")
DATA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
DELIVERY = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
DEVICE = "NVIDIA A100-PCIE-40GB"
ORDERS = ("full0","within0")
PUPIL_ARMS = ("hard","same_trial","cross_trial")
