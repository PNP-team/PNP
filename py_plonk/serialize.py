from .transcript import flags
import torch.nn.functional as F


def buffer_byte_size(modulus_bits):
    return (modulus_bits + 7) // 8

def write(content):
    bytes_content = bytearray()
    for partial in content.tolist():
        bytes_content.extend(partial.to_bytes(8, byteorder='little'))
    return bytes_content

def todo_serialize_with_flags(x, writer:list, flag = flags.EmptyFlags):
    assert flag.BIT_SIZE <= 8, "not enough space"
    x_base = F.to_base(x)
    bytes = write(x_base)
    bytes[- 1] |= flag.u8_bitmask()
    writer.extend(bytes)
    return writer
