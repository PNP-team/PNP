import math
from dataclasses import dataclass
from .serialize import buffer_byte_size
from .bytes import write
from .transcript import flags
import torch.nn.functional as F
import torch

@dataclass
class field:

    @classmethod
    def zero(cls):
        return torch.tensor([0 for _ in range(cls.Limbs)], dtype=cls.Dtype)
    
    # Return the Multiplicative identity
    @classmethod
    def one(cls):
        return cls.R.clone()

    @classmethod
    def make_tensor(cls, x): #x is a integer in base domain
        assert(x.bit_length() < 64)
        output = [x] + [0] * (cls.Limbs - 1)
        output = torch.tensor(output,dtype = cls.Base_type)
        return F.to_mont(output)

    # Returns the multiplicative generator of `char()` - 1 order.
    @classmethod
    def multiplicative_generator(cls):
        return cls.GENERATOR

    # Returns the root of unity of order n, if one exists.
    # If no small multiplicative subgroup is defined, this is the 2-adic root of unity of order n
    # (for n a power of 2).
    @classmethod
    def get_root_of_unity(cls,n):
        assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
        log_size_of_group = n.bit_length() - 1
        assert log_size_of_group <= cls.TWO_ADICITY, "logn must <= TWO_ADICITY"

        base = cls.TWO_ADIC_ROOT_OF_UNITY
        exponent = 1 << (cls.TWO_ADICITY - log_size_of_group)
        return F.exp_mod(base, exponent)

    def write(self,writer):
        content = F.to_base(self.value)
        new_writer = write(content,writer)
        return new_writer
    
    def serialize_with_flags(self, writer:list, flag:flags):
        if flag.BIT_SIZE > 8:
            print("Not enough space")
            return

        if flag.BIT_SIZE != 0:
            print("Flag size is not zero", flag.BIT_SIZE)

        output_byte_size = self.value.size()[-1] * 8
        # output_byte_size = buffer_byte_size(self.MODULUS_BITS + flag.BIT_SIZE)

        bytes = bytearray(self.BYTE_SIZE+1)
        modified_bytes = self.write(bytes[:self.BYTE_SIZE])
        bytes = modified_bytes+bytes[self.BYTE_SIZE:]
        bytes[output_byte_size - 1] |= flag.u8_bitmask()
        writer.extend(bytes[:output_byte_size])
        return writer
    
    def serialize(self,writer):
        writer = self.serialize_with_flags(writer, flags.EmptyFlags)
        return writer
    


