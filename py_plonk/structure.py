from dataclasses import dataclass
from typing import List
from .bls12_381 import fq
import torch


@dataclass
class G2Coordinate:
    c0: any
    c1: any


@dataclass
class AffinePointG1:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_zero(self):
        return torch.equal(self.x, fq.zero()) and torch.equal(self.y, fq.one())


@dataclass
class AffinePointG2:
    x: G2Coordinate
    y: G2Coordinate

@dataclass
class BTreeMap:
    def __init__(self, item, pos):
        self.item = item
        self.pos = pos

class UniversalParams:
    def __init__(self, powers_of_g, powers_of_gamma_g):
        self.powers_of_g = powers_of_g
        self.powers_of_gamma_g = powers_of_gamma_g


@dataclass
class OpenProof:
    # This is a commitment to the witness polynomial; see [KZG10] for more details.
    w: AffinePointG1
    # This is the evaluation of the random polynomial at the point for which
    # the evaluation proof was produced.
    random_v: any
