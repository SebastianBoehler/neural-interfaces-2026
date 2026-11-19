"""Exact parent releases and banks for the joint test."""
from pathlib import Path

BASE = Path('/mnt/lustre/work/gehler/gfh527')
CAMP = BASE / 'ni26_friday_causal_rebuild_20261118'
C20 = CAMP / 'fc20_outer_v1'
C21 = BASE / 'ni26_friday_c21_corrected_v2_20261119'
F25 = CAMP / 'f25c_v1'
PUPIL_SOURCE = 'b15e7155853a7694387deb618a4786f095b6fc26bbe282aec2129ec2ef68dcb5'
PINS = {
    'within0': {
        'fbl_proof': 'c41f6df763d86dee454db348560fce1943620f6ed3a3a1ef8824d4c54ac38472',
        'fbl_array': '109a572de0f062a62ddf69d80f847d5a107e33236b45f34d7984af303df6df45',
        'raw_proof': '9ed2daccdce0729d867f7af915ab50840e2a43340b6de9831f4e85485412aab1',
        'raw_array': '3da8bcc957b446a9dfa7b4d0c1c7a5ccf18b6feedc205f4d18dcd986460e3d79',
    },
    'full0': {
        'fbl_proof': '442ba868279c332e550bc4cffbd29f372c1e1afdf7fb8d6e86fe1a548fd6f331',
        'fbl_array': '5b9da9b5ba865c4b0b6bcdd2251536a472884d561bda85cfb4e624358f3f5650',
        'raw_proof': '9d2986fe5645cad5b45bec00380f43095ae221ed92e1b88664308a67e45fc4ec',
        'raw_array': '3e30eece73b68b94d33d57b9378bbcae4f3fcb295b2ee56b2ba7a415e68400bf',
    },
}
ARMS = ('stock', 'identity', 'FBL2', 'multi', 'joint', 'single_joint')
