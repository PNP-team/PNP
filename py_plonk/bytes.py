import struct
import torch
from .bls12_381 import fr

def write_new(content,writer:list):
    limbs = len(writer)//8
    content_part = [[] for _ in range(limbs)]
    bytes_content = [[] for _ in range(limbs)]
    content=content.tolist()
    for i in range(limbs):
        # content_part[i] = int(content & 0xFFFFFFFFFFFFFFFF)
        if content[i] == 0:
            bytes_content[i] = b'\x00' * 8  # 以8个字节的空字节填充
        else:
            bytes_content[i] = content[i].to_bytes(8, byteorder='little')
        # content = content >> 64

    little_endian_bytes = [item for sublist in bytes_content for item in sublist]
    byte_list = [byte for byte in little_endian_bytes]

    for i in range(len(writer)):
        writer[i] = byte_list[i]
    return writer




def read(reader):
    format_string = "<" + "Q" * fr.LIMBS()
    data = struct.unpack_from(format_string, reader)
    res = torch.tensor(data, dtype = fr.BASE_TYPE())
    return res