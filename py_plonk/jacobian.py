from .structure import AffinePointG1, ProjectivePointG1
from .bls12_381 import fq
import torch.nn.functional as F
import copy

COEFF_A = 0

def to_affine(input: ProjectivePointG1):

    px = input.x
    py = input.y
    pz = input.z

    if input.is_zero():
        x = fq.zero()
        y = fq.one()
        return AffinePointG1(x, y)
    else:
        # Z is nonzero, so it must have an inverse in a field.
        # div_mod work on cpu

        zinv = F.div_mod(fq.one(), pz)
        zinv_squared = F.mul_mod(zinv, zinv)

        x = F.mul_mod(px, zinv_squared)
        mid1 = F.mul_mod(zinv_squared, zinv)
        y = F.mul_mod(py, mid1)

        return AffinePointG1(x, y)

def double_ProjectivePointG1(self: ProjectivePointG1):

    if self.is_zero():
        return self

    if COEFF_A == 0:
        # A = X1^2
        a = F.mul_mod(self.x, self.x)

        # B = Y1^2
        b = F.mul_mod(self.y, self.y)

        # C = B^2
        c = F.mul_mod(b, b)

        # D = 2*((X1+B)^2-A-C)
        mid1 = F.add_mod(self.x, b)
        mid1 = F.mul_mod(mid1, mid1)
        mid2 = F.sub_mod(mid1, a)
        mid2 = F.sub_mod(mid2, c)
        d = F.add_mod(mid2, mid2)

        # E = 3*A
        mid1 = F.add_mod(a, a)
        e = F.add_mod(mid1, a)

        # F = E^2
        f = F.mul_mod(e, e)

        # Z3 = 2*Y1*Z1
        mid1 = F.mul_mod(self.y, self.z)
        z = F.add_mod(mid1, mid1)

        # X3 = F-2*D
        mid1 = F.sub_mod(f, d)
        x = F.sub_mod(mid1, d)

        # Y3 = E*(D-X3)-8*C
        mid1 = F.sub_mod(d, x)
        mid2 = F.add_mod(c, c)
        mid2 = F.add_mod(mid2, mid2)
        mid2 = F.add_mod(mid2, mid2)
        mid3 = F.mul_mod(e, mid1)
        y = F.sub_mod(mid3, mid2)

        return ProjectivePointG1(x, y, z)
    
def add_assign(self: ProjectivePointG1, other: ProjectivePointG1):

    if self.is_zero():
        return copy.deepcopy(other)

    if other.is_zero():
        return copy.deepcopy(self)

    # Z1Z1 = Z1^2
    z1z1 = F.mul_mod(self.z, self.z)

    # Z2Z2 = Z2^2
    z2z2 = F.mul_mod(other.z, other.z)

    # U1 = X1*Z2Z2
    u1 = F.mul_mod(self.x, z2z2)

    # U2 = X2*Z1Z1
    u2 = F.mul_mod(other.x, z1z1)

    # S1 = Y1*Z2*Z2Z2
    s1 = F.mul_mod(self.y, other.z)
    s1 = F.mul_mod(s1, z2z2)

    # S2 = Y2*Z1*Z1Z1
    s2 = F.mul_mod(other.y, self.z)
    s2 = F.mul_mod(s2, z1z1)

    if F.trace_equal(u1, u2) and F.trace_equal(s1, s2):
        # The two points are equal, so we double.
        return double_ProjectivePointG1(self)
    else:
        # H = U2-U1
        h = F.sub_mod(u2, u1)

        # I = (2*H)^2
        i = F.mul_mod(F.add_mod(h, h), F.add_mod(h, h))

        # J = H*I
        j = F.mul_mod(h, i)

        # r = 2*(S2-S1)
        r = F.add_mod(F.sub_mod(s2, s1), F.sub_mod(s2, s1))

        # V = U1*I
        v = F.mul_mod(u1, i)

        # X3 = r^2 - J - 2*V
        x = F.sub_mod(F.sub_mod(F.mul_mod(r, r), j), F.add_mod(v, v))

        # Y3 = r*(V - X3) - 2*S1*J
        y = F.sub_mod(
            F.mul_mod(r, F.sub_mod(v, x)), F.add_mod(F.mul_mod(s1, j), F.mul_mod(s1, j))
        )

        # Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
        z = F.mul_mod(
            F.sub_mod(
                F.sub_mod(
                    F.mul_mod(F.add_mod(self.z, other.z), F.add_mod(self.z, other.z)),
                    z1z1,
                ),
                z2z2,
            ),
            h,
        )

        return ProjectivePointG1(x, y, z)
