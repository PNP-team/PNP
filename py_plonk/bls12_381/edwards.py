import torch
from ..bls12_381 import fr


def COEFF_A():
    return torch.tensor(
        [
            18446744060824649731,
            18102478225614246908,
            11073656695919314959,
            6613806504683796440,
        ],
        dtype=fr.TYPE(),
    ).to("cuda")


def COEFF_D():
    return torch.tensor(
        [
            3049539848285517488,
            18189135023605205683,
            8793554888777148625,
            6339087681201251886,
        ],
        dtype=fr.TYPE(),
    ).to("cuda")
