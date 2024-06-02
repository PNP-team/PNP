from .transcript import flags
import torch
import torch.nn.functional as F
from .structure import AffinePointG1, BTreeMap
from .bls12_381 import fr, fq
import struct

def serialize(item, flag=flags.EmptyFlags):
    if isinstance(item, int):
        return item.to_bytes(8, byteorder="little")

    if isinstance(item, torch.Tensor):
        assert flag.BIT_SIZE <= 8, "not enough space"
        item_base = F.to_base(item)
        byte_array = bytearray()
        for partial in item_base.tolist():
            byte_array.extend(partial.to_bytes(8, byteorder="little"))
        byte_array[-1] |= flag.u8_bitmask()
        return byte_array

    if isinstance(item, BTreeMap):
        len = 1  # len = item.length
        return serialize(len) + serialize(item.pos) + serialize(item.item)

    if isinstance(item, AffinePointG1):
        if item.is_zero():
            flag = flags.SWFlags.infinity()
            return serialize(fq.zero(), flag)
        else:
            a = F.to_base(item.y)
            b = F.to_base(F.neg_mod(item.y))
            a_most_sig = a[-1].tolist()
            b_most_sig = b[-1].tolist()
            flag = flags.SWFlags.from_y_sign(a_most_sig > b_most_sig)
            return serialize(item.x, flag)

    assert False, "unsupported type"

def deserialize(x):
    assert flags.EmptyFlags.BIT_SIZE <= 8, "empty flags too large"

    aligned = bytearray(x)
    aligned.extend([0] * (8 - len(x) % 8))
    format_string = "<" + "Q" * fr.LIMBS()
    scalar_in_uint64 = struct.unpack_from(format_string, aligned)
    base_x = torch.tensor(scalar_in_uint64, dtype=fr.BASE_TYPE())
    return F.to_mont(base_x)