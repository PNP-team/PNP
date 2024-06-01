import math
from dataclasses import dataclass
from .serialize import buffer_byte_size
from .bytes import write
from .transcript import flags
import torch.nn.functional as F
import torch

@dataclass
class field:
    pass
    # @classmethod
    # def zero(cls):
    #     return torch.tensor([0 for _ in range(cls.Limbs)], dtype=cls.Dtype)
    
    # # Return the Multiplicative identity
    # @classmethod
    # def one(cls):
    #     return cls.R.clone()



    # def write(self,writer):
    #     content = F.to_base(self.value)
    #     new_writer = write(content,writer)
    #     return new_writer
    
    # def serialize_with_flags(self, writer:list, flag:flags):
    #     if flag.BIT_SIZE > 8:
    #         print("Not enough space")
    #         return

    #     if flag.BIT_SIZE != 0:
    #         print("Flag size is not zero", flag.BIT_SIZE)

    #     # output_byte_size = self.value.size()[-1] * 8
    #     output_byte_size = buffer_byte_size(self.MODULUS_BITS() + flag.BIT_SIZE)

    #     bytes = bytearray(self.BYTE_SIZE()+1)
    #     modified_bytes = self.write(bytes[:self.BYTE_SIZE()])
    #     bytes = modified_bytes+bytes[self.BYTE_SIZE():]
    #     bytes[output_byte_size - 1] |= flag.u8_bitmask()
    #     writer.extend(bytes[:output_byte_size])
    #     return writer
    
    # def serialize(self,writer):
    #     writer = self.serialize_with_flags(writer, flags.EmptyFlags)
    #     return writer
    


