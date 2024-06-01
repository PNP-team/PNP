import torch


def TYPE():
    return torch.BLS12_381_Fq_G1_Mont


def MODULUS_BITS():
    return 381


def BYTE_SIZE():
    return (381 + 7) // 8


# def LIMBS():
#     return (381 + 63) // 64


def zero():
    return torch.tensor([0, 0, 0, 0, 0, 0], dtype=TYPE())


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
        dtype=TYPE(),
    )

