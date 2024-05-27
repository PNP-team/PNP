import struct
import torch

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


def write(content,writer:list):
    limbs = len(writer)//8
    bytes_content = [[] for _ in range(limbs)]
    
    for i in range(limbs):
        partial = int(content[i].tolist())
        if partial == 0:
            bytes_content[i] = b'\x00' * 8  # 以8个字节的空字节填充
        else:
            bytes_content[i] = partial.to_bytes(8, byteorder='little')

    little_endian_bytes = [item for sublist in bytes_content for item in sublist]
    for i in range(len(writer)):
        writer[i] = little_endian_bytes[i]
    return writer

def read(reader,cls):
    format_string = "<" + "Q" * cls.Limbs
    data = struct.unpack_from(format_string, reader)
    res = torch.tensor(data, dtype = cls.Base_type)
    return res