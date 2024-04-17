from .....bls12_381 import fr
from dataclasses import dataclass
from .....arithmetic import from_gmpy_list_1
import torch
import  torch.nn.functional as F
from .....arithmetic import extend_tensor
@dataclass
class WitnessValues:
    a_val: fr.Fr  # Left Value
    b_val: fr.Fr  # Right Value
    c_val: fr.Fr  # Output Value
    d_val: fr.Fr  # Fourth Value


def delta(f:fr.Fr,size):
    
    one = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    two = fr.Fr.from_repr(2)
    three = fr.Fr.from_repr(3)
    two = torch.tensor(from_gmpy_list_1(two),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    three = torch.tensor(from_gmpy_list_1(three),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')

    if size ==0 :
        pass
    else :
        one =extend_tensor(one,size)
        two= extend_tensor(two,size)
        three = extend_tensor(three ,size)
    f_1 = F.sub_mod(f, one)
    f_2 = F.sub_mod(f, two)
    f_3 = F.sub_mod(f, three)
    mid1 = F.mul_mod(f_1, f_2)
    mid2 = F.mul_mod(mid1, f_3)
    res = F.mul_mod(f, mid2)

    
    return res