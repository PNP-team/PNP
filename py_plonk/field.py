import gmpy2
import math
from dataclasses import dataclass
from .serialize import buffer_byte_size
from .bytes import write
from .transcript import flags
import torch.nn.functional as F
import torch

def to_tensor_base(input, type):
    output = []
    for j in range(type.Limbs):
        output.append( int(input & 0xFFFFFFFFFFFFFFFF) )
        input = input >> 64
    output = torch.tensor(output,dtype = type.Base_type)
    return output

def to_tensor_mont(input, type):
    output = []
    for j in range(type.Limbs):
        output.append( int(input & 0xFFFFFFFFFFFFFFFF) )
        input = input >> 64
    output = torch.tensor(output,dtype = type.Dtype)
    return output

@dataclass
class field:

    @classmethod
    def zero(cls):
        return cls(torch.tensor([0 for _ in range(cls.Limbs)], dtype=cls.Dtype))
    # def zero(cls):
    #     return cls(gmpy2.mpz(0))
    
    # Return the Multiplicative identity
    @classmethod
    def one(cls):
        # cls = type(self)
        return cls(cls.R)
    
    def add(self,b):
        cls = type(self)
        res = self.value + b.value
        res %= cls.MODULUS
        return cls(res)
    
    def sub(self,b):
        cls = type(self)
        res = F.sub_mod(self.value, b.value)
        return cls(res)
    
    def neg(self):
        cls = type(self)
        if torch.equal(self.value, cls.zero().value) :
            return self
        else:
            temp = cls(cls.MODULUS)
            res = temp.sub(self)
            return res
        
    def double(self):
        res = self.add(self)
        return res

    #mongomery mul
    def mul(self, b):
        cls = type(self)
        res = F.mul_mod(self.value, b.value)
        return cls(res)
    
    def square(self):
        # self = F.mul_mod(self)
        self = self.mul(self)
        return self
    
    def pow(self,exp):
        res = self.value.clone()
        cls = type(self)
        for i in range(exp):
            #modsquare
            res = F.mul_mod(res, res)
        return cls(res)
    
    # def pow(self,exp):
    #     res = self.one()
    #     for i in range(63,-1,-1):
    #         #modsquare
    #         res = res.square()
    #         if ((exp >> i) & 1) == 1:
    #             res = res.mul(self)
    #             # res = F.mul_mod(self)
    #     return res
    
    # ##TODO 暂时用于quotient函数
    # def pow_1(self,exp):
            
    #     def square_1(self):
    #         self = F.mul_mod(self)
    #         # self = self.mul(self)
    #         return self
    #     res = self.clone()
    #     for i in range(63,-1,-1):
    #         #modsquare
    #         res = res.square_1()
    #         if ((exp >> i) & 1) == 1:
    #             # res = res.mul(self)
    #             res = F.mul_mod(self)
    #     return res
    
    @classmethod
    def from_repr(cls, r):
        if(type(r) == int):
            r = to_tensor_base(r,cls)
        r = F.to_mont(r)
        return cls(r)
    # @classmethod
    # def from_repr(cls, r):
    #     if r == 0:
    #         return cls(r)
    #     else:  
    #         r=cls(r)
    #         R2 = cls(cls.R2)
    #         r = r.mul(R2)
    #         return r
        
    #Montgomery Reduction
    def into_repr(self):
        # self = to_tensor_mont(self.value, type(self))
        res = F.to_base(self.value)
        return res
        
    # def into_repr(self):
    #     if self.value == 0:
    #         return self.value
    #     else:
    #         res = self.value * self.R_INV
    #         res %= self.MODULUS
    #         return res

    # @classmethod
    # def inverse(cls,self,cls):
    #     if self==0:
    #         print("cannot invert 0!\n")
    #         return  None
    #     #judge if self is already in field
    #     elif type(self)==field:
    #         x = field(self.value,self.cls) 
    #         x.value = gmpy2.invert(x.value, cls.MODULUS)
    #         R2 = field(cls.R2,cls)
    #         x = x.mul(R2)
    #         x = x.mul(R2)
    #         return x
    #     else:
    #         x=field.from_repr(self,cls)
    #         x.value = gmpy2.invert(x.value, cls.MODULUS)
    #         R2 = field(cls.R2,cls)
    #         x = x.mul(R2)
    #         x = x.mul(R2)
    #         return x
    @classmethod
    def inverse(cls,self):
        if torch.equal(self.value, cls.zero().value):
             print("cannot invert zero!\n")
             return  None
        one = cls.one()
        res = F.div_mod(one.value, self.value)
        return cls(res)
    
    # def inverse(cls,self):
    #     if self == 0:
    #          print("cannot invert 0!\n")
    #          return  None
    #     u = self
    #     if type(self) != gmpy2.mpz:
    #         u = self.value
    #     one =gmpy2.mpz(1)
    #     v = cls.MODULUS
    #     b = cls.R2
    #     c = gmpy2.mpz(0)

    #     while u != one and v != one:
    #         while u & 1 == 0:
    #             u = u // 2
    #             if b & 1 == 0:
    #                 b = b // 2
    #             else:
    #                 b = b + cls.MODULUS
    #                 b = b // 2
    #         while v & 1 == 0:
    #             v =v // 2
    #             if c & 1 == 0:
    #                 c = c // 2
    #             else:
    #                 c = c + cls.MODULUS
    #                 c = c // 2
    #         if v < u:
    #             u = u-v
    #             if c > b:
    #                 b = b + cls.MODULUS
    #             b = b - c
    #             b = gmpy2.f_mod(b, cls.MODULUS)
    #         else:
    #             v = v-u
    #             if b > c:
    #                 c = c + cls.MODULUS
    #             c = c - b
    #             c = gmpy2.f_mod(c, cls.MODULUS)
    #     if u == one:
    #         return cls(b)
    #     else:
    #         return cls(c)
    
    # Returns the 2^s root of unity.
    @classmethod
    def two_adic_root_of_unity(cls):
        return cls.TWO_ADIC_ROOT_OF_UNITY 

    # Returns the 2^s * small_subgroup_base^small_subgroup_base_adicity root of unity
    # if a small subgroup is defined.
    def large_subgroup_root_of_unity():
        pass

    # Returns the multiplicative generator of `char()` - 1 order.
    @classmethod
    def multiplicative_generator(cls):
        return cls(cls.GENERATOR)

    # Returns the root of unity of order n, if one exists.
    # If no small multiplicative subgroup is defined, this is the 2-adic root of unity of order n
    # (for n a power of 2).
    @classmethod
    def get_root_of_unity(cls,n):
        size = 2 ** (n.bit_length()-1)
        log_size_of_group = int(math.log2(size))

        if n != size or log_size_of_group > cls.TWO_ADICITY:
            return None

        # Compute the generator for the multiplicative subgroup.
        # It should be 2^(log_size_of_group) root of unity.
        omega = cls.two_adic_root_of_unity()
        # R_inv=gmpy2.invert(cls.R,cls.MODULUS)
        for _ in range(log_size_of_group, cls.TWO_ADICITY):
            #modsquare
            omega = F.mul_mod(omega,omega)
            # omega *=omega
            # omega *=R_inv
            # omega %=cls.MODULUS
            
        # return torch.tensor(from_gmpy_list_1(omega),dtype=torch.BLS12_381_Fr_G1_Mont)
        return cls(omega)
            

    def write(self,writer):
        content = self.into_repr()
        new_writer = write(content,writer)
        return new_writer
    
    def serialize_with_flags(self, writer:list, flag:flags):
        if flag.BIT_SIZE > 8:
            print("Not enough space")
            return

        output_byte_size = buffer_byte_size(self.MODULUS_BITS + flag.BIT_SIZE)

        bytes = bytearray(self.BYTE_SIZE+1)
        modified_bytes = self.write(bytes[:self.BYTE_SIZE])
        bytes = modified_bytes+bytes[self.BYTE_SIZE:]
        bytes[output_byte_size - 1] |= flag.u8_bitmask()
        writer.extend(bytes[:output_byte_size])
        return writer
    
    def serialize(self,writer):
        writer = self.serialize_with_flags(writer,flags.EmptyFlags)
        return writer
    


        