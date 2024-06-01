from .transcript import flags
import torch.nn.functional as F
from .bytes import write


def buffer_byte_size(modulus_bits):
    return (modulus_bits + 7) // 8

def this_write(x,writer):
    content = F.to_base(x)
    new_writer = write(content,writer)
    return new_writer

def todo_serialize_with_flags(x, writer:list, flag:flags):
    if flag.BIT_SIZE > 8:
        print("Not enough space")
        return

    output_byte_size = x.size()[-1] * 8 # buffer_byte_size(self.MODULUS_BITS() + flag.BIT_SIZE)

    bytes = bytearray(output_byte_size+1)
    modified_bytes = this_write(x, bytes[:output_byte_size])
    bytes = modified_bytes+bytes[output_byte_size:]
    bytes[output_byte_size - 1] |= flag.u8_bitmask()
    writer.extend(bytes[:output_byte_size])
    return writer

def todo_serialize(x,writer):
    writer = todo_serialize_with_flags(x, writer, flags.EmptyFlags)
    return writer