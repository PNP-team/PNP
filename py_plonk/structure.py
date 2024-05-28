from dataclasses import dataclass
from typing import List
from .bls12_381 import fq
from .transcript import flags
import torch

@dataclass
class G2Coordinate:
    c0: any
    c1: any

@dataclass
class AffinePointG1:
    x: fq.Fq
    y: fq.Fq

    @classmethod
    def new(cls,x,y):
        return cls(x,y)

    #Returns the point at infinity, which always has Z = 0.
    @classmethod
    def zero(cls):
        x=fq.Fq.zero()
        y=x.one()
        return cls(x,y)
    
    def is_zero(self):
        one = fq.Fq.one()
        return torch.equal(self.x.value, fq.Fq.zero())  and torch.equal(one, self.y.value)

    
    def serialize(self,writer):
        if self.is_zero():
            flag = flags.SWFlags.infinity()
            zero = fq.Fq(fq.Fq.zero())
            writer = zero.serialize_with_flags(writer,flag)
            return writer
        else:
            neg_y = self.y.neg()
            a = self.y.into_repr()
            b = neg_y.into_repr()
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





