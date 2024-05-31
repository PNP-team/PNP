import torch

TYPE = torch.BLS12_381_Fq_G1_Mont
MODULUS_BITS = 381
BYTE_SIZE = (MODULUS_BITS + 7) // 8
LIMBS = (MODULUS_BITS + 63) // 64


def zero():
    return torch.tensor([0, 0, 0, 0, 0, 0], dtype=TYPE)


def one():
    return torch.tensor(
        [
            8505329371266088957,
            17002214543764226050,
            6865905132761471162,
            8632934651105793861,
            6631298214892334189,
            1582556514881692819,
        ],
        dtype=TYPE,
    )


# class Fq(field):
#     def __init__(self, value: torch.Tensor):
#         self.value = value

#     MODULUS_BITS: int = 381

#     # # 384bits
    # BYTE_SIZE: int = 48


# FQ_ONE = 1
# FQ_ZERO = 0
# COEFF_A = 0
# COEFF_B = 4
