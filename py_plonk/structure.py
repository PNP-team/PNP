from dataclasses import dataclass
from typing import List
from .bls12_381 import fq
from .transcript import flags
import torch
import torch.nn.functional as F
from .serialize import serialize_with_flags


@dataclass
class G2Coordinate:
    c0: any
    c1: any

@dataclass
class AffinePointG1:

    def __init__(self,x,y):
        self.x = x
        self.y = y


    @classmethod
    def new(cls,x,y):
        return cls(x,y)

    #Returns the point at infinity, which always has Z = 0.
    @classmethod
    def zero(cls):
        x=fq.zero()
        y=x.one()
        return cls(x,y)
    
    def is_zero(self):
        one = fq.one()
        return F.trace_equal(self.x.value, fq.zero())  and F.trace_equal(one, self.y.value)

    
    def serialize(self,writer):
        if self.is_zero():
            flag = flags.SWFlags.infinity()
            # zero = fq.Fq(fq.zero())
            writer = serialize_with_flags(fq.zero(), writer,flag) # zero.serialize_with_flags(writer,flag)
            return writer
        else:
            a = F.to_base(self.y.value)
            b = F.to_base(F.neg_mod(self.y.value))
            a_most_sig = a[-1].tolist()
            b_most_sig = b[-1].tolist()
            flag = flags.SWFlags.from_y_sign(a_most_sig > b_most_sig) #a > b
            writer = self.x.serialize_with_flags(writer, flag)
            return writer
@dataclass
class AffinePointG2:
    x: G2Coordinate
    y: G2Coordinate
#TODO: implement BTreeMap class
#TODO: implement different `serialize` for different types
def serialize_u64(item, writer: list):
    bytes = item.to_bytes(8, byteorder='little')
    writer.extend(bytes)
    return writer

def serialize_BTreeMap(item, pos, writer: list):
    len = 1    # len = item.length
    writer = serialize_u64(len, writer)
    writer = serialize_u64(pos, writer)
    writer = item.serialize(writer)
    return writer

@dataclass
class UniversalParams:
    powers_of_g: List[AffinePointG1]
    powers_of_gamma_g: List[AffinePointG1]
    h: any
    beta_h: any


@dataclass
class OpenProof:
    # This is a commitment to the witness polynomial; see [KZG10] for more details.
    w: AffinePointG1
    # This is the evaluation of the random polynomial at the point for which
    # the evaluation proof was produced.
    random_v: any





