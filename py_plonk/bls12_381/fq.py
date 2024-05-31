import torch
from dataclasses import dataclass
from ..field import field


class Fq(field):
    def __init__(self, value: torch.Tensor):
        self.value = value
    
    Dtype = torch.BLS12_381_Fq_G1_Mont

    Limbs: int = 6

    MODULUS_BITS: int = 381

    R = torch.tensor([ 8505329371266088957, 17002214543764226050,  6865905132761471162,
         8632934651105793861,  6631298214892334189,  1582556514881692819],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    #384bits
    BYTE_SIZE: int = 48


# FQ_ONE = 1
# FQ_ZERO = 0
# COEFF_A = 0
# COEFF_B = 4