import torch
import torch.nn.functional as F

class Fr:
    pass

def TYPE():
    return torch.BLS12_381_Fr_G1_Mont


def BASE_TYPE():
    return torch.BLS12_381_Fr_G1_Base


def MODULUS_BITS():
    return 255


def BYTE_SIZE():
    return (255 + 7) // 8


def LIMBS():
    return (255 + 63) // 64


def TWO_ADICITY():
    return 32


def zero():
    return torch.tensor([0, 0, 0, 0], dtype=TYPE())


def one():
    return torch.tensor(
        [
            8589934590,
            6378425256633387010,
            11064306276430008309,
            1739710354780652911,
        ],
        dtype=TYPE(),
    )


def MODULUS():
    return torch.tensor(
        [
            18446744069414584321,
            6034159408538082302,
            3691218898639771653,
            8353516859464449352,
        ],
        dtype=TYPE(),
    )


def TWO_ADIC_ROOT_OF_UNITY():
    return torch.tensor(
        [
            13381757501831005802,
            6564924994866501612,
            789602057691799140,
            6625830629041353339,
        ],
        dtype=TYPE(),
    )


def GENERATOR():
    return torch.tensor(
        [64424509425, 1721329240476523535, 18418692815241631664, 3824455624000121028],
        dtype=TYPE(),
    )


def REPR_SHAVE_BITS():
    return 1


def make_tensor(x, n=1):  # x is a integer in base domain
    assert x.bit_length() < 64
    output = [x] + [0] * (LIMBS() - 1)
    output = torch.tensor(output * n, dtype=BASE_TYPE())
    return F.to_mont(output)


