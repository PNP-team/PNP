from .....bls12_381 import fr

import torch
import torch.nn.functional as F


class WitnessValues:
    def __init__(self, a_val, b_val, c_val, d_val):
        self.a_val = a_val
        self.b_val = b_val
        self.c_val = c_val
        self.d_val = d_val


def delta(f: torch.Tensor):

    one = fr.one()
    two = fr.make_tensor(2)
    three = fr.make_tensor(3)

    f_1 = F.sub_mod_scalar(f, one.to("cuda"))
    f_2 = F.sub_mod_scalar(f, two.to("cuda"))
    f_3 = F.sub_mod_scalar(f, three.to("cuda"))
    mid1 = F.mul_mod(f_1, f_2)
    mid2 = F.mul_mod(mid1, f_3)
    res = F.mul_mod(f, mid2)

    return res
