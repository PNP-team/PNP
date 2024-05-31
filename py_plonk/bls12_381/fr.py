import torch
from ..transcript import flags
from ..serialize import buffer_byte_size
from ..bytes import read
import torch.nn.functional as F

TYPE = torch.BLS12_381_Fr_G1_Mont
MODULUS_BITS = 255
BYTE_SIZE = (MODULUS_BITS + 7) // 8
LIMBS = (MODULUS_BITS + 63) // 64

TWO_ADICITY = 32


def zero():
    return torch.tensor([0, 0, 0, 0], dtype=TYPE)


def one():
    return torch.tensor(
        [
            8589934590,
            6378425256633387010,
            11064306276430008309,
            1739710354780652911,
        ],
        dtype=TYPE,
    )


def MODULUS():
    return torch.tensor(
        [
            18446744069414584321,
            6034159408538082302,
            3691218898639771653,
            8353516859464449352,
        ],
        dtype=TYPE,
    )


def TWO_ADIC_ROOT_OF_UNITY():
    return torch.tensor(
        [
            13381757501831005802,
            6564924994866501612,
            789602057691799140,
            6625830629041353339,
        ],
        dtype=TYPE,
    )


def GENERATOR():
    return torch.tensor(
        [64424509425, 1721329240476523535, 18418692815241631664, 3824455624000121028],
        dtype=TYPE,
    )


class Fr:
    def __init__(self, value: torch.Tensor):
        self.value = value

    # Dtype = torch.BLS12_381_Fr_G1_Mont

    Base_type = torch.BLS12_381_Fr_G1_Base

    Limbs: int = 4
    MODULUS_BITS: int = 255
    REPR_SHAVE_BITS: int = 1


    # 256bits
    BYTE_SIZE: int = 32

    @classmethod
    def make_tensor(cls, x):  # x is a integer in base domain
        assert x.bit_length() < 64
        output = [x] + [0] * (cls.Limbs - 1)
        output = torch.tensor(output, dtype=cls.Base_type)
        return F.to_mont(output)


def deserialize(reader):
    output_byte_size = buffer_byte_size(Fr.MODULUS_BITS + flags.EmptyFlags.BIT_SIZE)

    masked_bytes = bytearray([0] * (Fr.BYTE_SIZE + 1))
    masked_bytes[:output_byte_size] = reader[:output_byte_size]

    flag = flags.EmptyFlags.from_u8_remove_flags(masked_bytes[output_byte_size - 1])

    element = read(masked_bytes, Fr)
    field_element = F.to_mont(element)
    return field_element, flag


def from_random_bytes(bytes: bytes):
    limbs = (len(bytes) + 1) // 8
    if flags.EmptyFlags.BIT_SIZE > 8:
        return None

    result_bytes = bytearray([0] * (limbs * 8 + 1))
    result_bytes[: len(bytes)] = bytes
    last_bytes_mask = bytearray(9)
    last_limb_mask = ((2**64 - 1) >> Fr.REPR_SHAVE_BITS).to_bytes(8, byteorder="little")
    last_bytes_mask[:8] = last_limb_mask[:]
    output_byte_size = buffer_byte_size(Fr.MODULUS_BITS + flags.EmptyFlags.BIT_SIZE)
    flag_location = output_byte_size - 1
    flag_location_in_last_limb = flag_location - (8 * (limbs - 1))

    last_bytes = result_bytes[8 * (limbs - 1) :]

    flags_mask = 0xFF >> (8 - flags.EmptyFlags.BIT_SIZE)
    flag = 0
    for i, (b, m) in enumerate(zip(last_bytes, last_bytes_mask)):
        if i == flag_location_in_last_limb:
            flag = b & flags_mask
        b &= m

    field_element, flag = deserialize(result_bytes[: limbs * 8])
    # flags_obj = flags.SWFlags.from_u8(flag)

    return field_element
