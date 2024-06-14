from dataclasses import dataclass
from .bls12_381 import fq
import torch

class AffinePointG1:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_zero(self):
        return torch.equal(self.x, fq.zero()) and torch.equal(self.y, fq.one())

class ProjectivePointG1:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def is_zero(self):
        return all(x == 0 for x in self.z.tolist())

class BTreeMap:
    def __init__(self, item, pos):
        self.item = item
        self.pos = pos

@dataclass
class OpenProof:
    # This is a commitment to the witness polynomial; see [KZG10] for more details.
    w: AffinePointG1
    # This is the evaluation of the random polynomial at the point for which
    # the evaluation proof was produced.
    random_v: any
