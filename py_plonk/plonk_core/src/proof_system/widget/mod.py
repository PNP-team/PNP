from .....bls12_381 import fr
from dataclasses import dataclass
import torch
import torch.nn.functional as F
@dataclass
class WitnessValues:
    a_val: fr.Fr  # Left Value
    b_val: fr.Fr  # Right Value
    c_val: fr.Fr  # Output Value
    d_val: fr.Fr  # Fourth Value


def delta(f: torch.Tensor):
    
    one = fr.Fr.one()
    two = fr.Fr.make_tensor(2)
    three = fr.Fr.make_tensor(3)

    f_1 = F.sub_mod_scalar(f, one.to("cuda"))
    f_2 = F.sub_mod_scalar(f, two.to("cuda"))
    f_3 = F.sub_mod_scalar(f, three.to("cuda"))
    mid1 = F.mul_mod(f_1, f_2)
    mid2 = F.mul_mod(mid1, f_3)
    res = F.mul_mod(f, mid2)

    
    return res