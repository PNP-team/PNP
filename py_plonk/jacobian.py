from dataclasses import dataclass
from typing import List
from .structure import AffinePointG1
from .bls12_381 import fq
import torch
import torch.nn.functional as F
import copy
# from .transcript import flags
# from .serialize import buffer_byte_size
# from .arithmetic import neg_fq
# from .ele import into_repr_fq
# from .bytes import write
COEFF_A=0
@dataclass

class ProjectivePointG1: 
    x: fq.Fq
    y: fq.Fq
    z: fq.Fq

    @classmethod
    #Returns the point at infinity, which always has Z = 0.
    def zero(cls):
        # fq R=3380320199399472671518931668520476396067793891014375699959770179129436917079669831430077592723774664465579537268733
        # x=field_elemnt.one()
        x= torch.tensor([8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861],dtype=torch.BLS12_381_Fr_G1_Mont)
        y=torch.tensor([8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861],dtype=torch.BLS12_381_Fr_G1_Mont)
        z=torch.zeros(1,4,dtype=torch.BLS12_381_Fr_G1_Mont)
        return cls(x,y,z)
    
    def is_zero(self):
        return torch.equal(self.z.value,torch.zeros(6,dtype=torch.BLS12_381_Fq_G1_Mont)) 
    
    def double(self):
        if self.is_zero():
            return self

        if fq.COEFF_A == 0:
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

        # else:
        #     # http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
        #     # XX = X1^2
        #     xx = self.x.square()

        #     # YY = Y1^2
        #     yy = self.y.square()

        #     # YYYY = YY^2
        #     yyyy = yy.square()

        #     # ZZ = Z1^2
        #     zz = self.z.square()

        #     # S = 2*((X1+YY)^2-XX-YYYY)
        #     mid1 = self.x.add(yy)
        #     mid1 = mid1.square()
        #     mid2 = mid1.sub(xx)
        #     mid1 = mid2.sub(yyyy)
        #     s = mid1.double()

        #     # M = 3*XX+a*ZZ^2
        #     mid1 = xx.double()
        #     mid1 = mid1.add(xx)
        #     mid2 = zz.square()

        #     m = xx + xx + xx + P.mul_by_a(zz.square())

        #     # T = M^2-2*S
        #     t = m.square() - s.double()

        #     # X3 = T
        #     self.x = t
        #     # Y3 = M*(S-T)-8*YYYY
        #     old_y = self.y
        #     self.y = m * (s - t) - yyyy.double_in_place().double_in_place().double_in_place()
        #     # Z3 = (Y1+Z1)^2-YY-ZZ
        #     self.z = (old_y + self.z).square() - yy - zz
        #     return self

    def to_affine(p: 'ProjectivePointG1'):
        # one = p.z.one()
        one = torch.tensor([8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861],dtype=torch.BLS12_381_Fr_G1_Mont)
        if p.is_zero():
            return AffinePointG1.zero()
        elif torch.equal(p.z , one):
            # If Z is one, the point is already normalized.
            return AffinePointG1.new(p.x, p.y)
        else:
            # Z is nonzero, so it must have an inverse in a field.
            #div_mod work on cpu
            zinv = F.div_mod(torch.tensor([8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861],dtype=torch.BLS12_381_Fr_G1_Mont),p.z)
            zinv_squared = F.mul_mod(zinv, zinv)

            x = F.mul_mod(p.x, zinv_squared)
            mid1 = F.mul_mod(zinv_squared, zinv)
            y = F.mul_mod(p.y, mid1)
            return AffinePointG1.new(x, y)

    def add_assign(self, other: 'ProjectivePointG1'):
        if self.is_zero():
            x, y, z = other.x, other.y, other.z
            return ProjectivePointG1(x, y, z)

        if other.is_zero():
            return ProjectivePointG1(self.x, self.y, self.z)

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

        if u1.value == u2.value and s1.value == s2.value:
            # The two points are equal, so we double.
            return self.double()
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
            y = F.sub_mod(F.mul_mod(r, F.sub_mod(v, x)), F.add_mod(F.mul_mod(s1, j), F.mul_mod(s1, j)))

            # Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
            z = F.mul_mod(F.sub_mod(F.sub_mod(F.mul_mod(F.add_mod(self.z, other.z), F.add_mod(self.z, other.z)), z1z1), z2z2), h)
            return ProjectivePointG1(x, y, z)

    def add_assign_mixed(self, other: 'AffinePointG1'):
        if other.is_zero():
            return ProjectivePointG1(self.x, self.y, self.z)

        elif self.is_zero():
            # If self is zero, return the other point in projective coordinates.
            x = other.x
            y = other.y
            #z = self.z.one()  # Assuming z.one() is a method to get a representation of one.
            z=  torch.tensor([8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861],dtype=torch.BLS12_381_Fr_G1_Mont)
            return ProjectivePointG1(x, y, z)
        else:
            # Z1Z1 = Z1^2
            z1z1 = F.mul_mod(self.z, self.z)

            # U2 = X2*Z1Z1
            u2 = F.mul_mod(other.x, z1z1)

            # S2 = Y2*Z1*Z1Z1
            s2 = F.mul_mod(other.y, self.z)
            s2 = F.mul_mod(s2, z1z1)

            if self.x.value == u2.value and self.y.value == s2.value:
                # The two points are equal, so we double.
                return self.double()
            else:
                # H = U2-X1
                h = F.sub_mod(u2, self.x)

                # I = 4*(H^2)
                i = F.mul_mod(h, h)
                i = F.add_mod(i, i)
                i = F.add_mod(i, i)

                # J = H*I
                j = F.mul_mod(h, i)

                # r = 2*(S2-Y1)
                r = F.sub_mod(s2, self.y)
                r = F.add_mod(r, r)

                # V = X1*I
                v = F.mul_mod(self.x, i)

                # X3 = r^2 - J - 2*V
                x = F.mul_mod(r, r)
                x = F.sub_mod(x, j)
                v2 = F.add_mod(v, v)
                x = F.sub_mod(x, v2)

                # Y3 = r*(V-X3) - 2*Y1*J
                y = F.sub_mod(v, x)
                y = F.mul_mod(r, y)
                s1j = F.mul_mod(self.y, j)
                s1j2 = F.add_mod(s1j, s1j)
                y = F.sub_mod(y, s1j2)

                # Z3 = (Z1+H)^2 - Z1Z1 - H^2
                z = F.add_mod(self.z, h)
                z = F.mul_mod(z, z)
                z = F.sub_mod(z, z1z1)
                hh = F.mul_mod(h, h)
                z = F.sub_mod(z, hh)

                return ProjectivePointG1(x, y, z) 
                
def is_zero_ProjectivePointG1(self):
    return torch.equal(self[2],torch.zeros(6,dtype=torch.BLS12_381_Fq_G1_Mont)) ##z

def is_zero_AffinePointG1(self):
        one = torch.tensor([8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861, 6631298214892334189, 1582556514881692819],dtype=torch.BLS12_381_Fq_G1_Mont)
        return torch.equal(self[0] ,torch.zeros(6,dtype=torch.BLS12_381_Fq_G1_Mont) )and  torch.equal(self[1] ,one) # x=0 y=one

def to_affine(input: ProjectivePointG1): 
        px = input.x.value.clone()
        py = input.y.value.clone()
        pz = input.z.value.clone()
        # p[0]:x p[1]:y p[2]:z
        one = fq.Fq.one()
        if input.is_zero():
            x = fq.Fq.zero().value
            y = one.value
            return AffinePointG1(fq.Fq(x), fq.Fq(y))

        else:
            # Z is nonzero, so it must have an inverse in a field.
            #div_mod work on cpu
            zinv = F.div_mod(one.value, pz)
            zinv_squared = F.mul_mod(zinv, zinv)

            x = F.mul_mod(px, zinv_squared)
            mid1 = F.mul_mod(zinv_squared, zinv)
            y = F.mul_mod(py, mid1)
         
            return AffinePointG1(fq.Fq(x),fq.Fq(y))
def add_assign(self, other: 'ProjectivePointG1'):
    if is_zero_ProjectivePointG1(self):
        x, y, z = other[0], other[1], other[2]
        return [x,y,z]

    if is_zero_ProjectivePointG1(other):
        return [self[0],self[1],self[2]]

    # Z1Z1 = Z1^2
    z1z1 = F.mul_mod(self[2], self[2])

    # Z2Z2 = Z2^2
    z2z2 = F.mul_mod(other[2], other[2])

    # U1 = X1*Z2Z2
    u1 = F.mul_mod(self[0], z2z2)

    # U2 = X2*Z1Z1
    u2 = F.mul_mod(other[0], z1z1)

    # S1 = Y1*Z2*Z2Z2
    s1 = F.mul_mod(self[1], other.z)
    s1 = F.mul_mod(s1, z2z2)
    
    # S2 = Y2*Z1*Z1Z1
    s2 = F.mul_mod(other[1], self[2])
    s2 = F.mul_mod(s2, z1z1)

    if  torch.equal(u1 ,u2)and torch.equal(s1 ,s2):
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
        y = F.sub_mod(F.mul_mod(r, F.sub_mod(v, x)), F.add_mod(F.mul_mod(s1, j), F.mul_mod(s1, j)))

        # Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
        z = F.mul_mod(F.sub_mod(F.sub_mod(F.mul_mod(F.add_mod(self[2], other[2]), F.add_mod(self[2], other[2])), z1z1), z2z2), h)
        # return ProjectivePointG1(x, y, z)
        return [x,y,z]

def double_ProjectivePointG1(self: ProjectivePointG1):
        if self.is_zero():
            return self

        if COEFF_A == 0:
            # A = X1^2
            a = F.mul_mod(self.x.value, self.x.value)

            # B = Y1^2
            b = F.mul_mod(self.y.value, self.y.value)

            # C = B^2
            c = F.mul_mod(b, b)

            # D = 2*((X1+B)^2-A-C)
            mid1 = F.add_mod(self.x.value, b)
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
            mid1 = F.mul_mod(self.y.value, self.z.value)
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

            # return ProjectivePointG1(x, y, z)
            return ProjectivePointG1(fq.Fq(x), fq.Fq(y), fq.Fq(z))

def add_assign_mixed(self1: ProjectivePointG1, other: 'AffinePointG1'):
    self = copy.deepcopy(self1)
    if  other.is_zero():
        # return ProjectivePointG1(self.x, self.y, self.z)
        output= copy.deepcopy(self1)
        return output

    elif self1.is_zero():
        # If self is zero, return the other point in projective coordinates.
        x = copy.deepcopy(other.x)
        y = copy.deepcopy(other.y)
        #z = self.z.one()  # Assuming z.one() is a method to get a representation of one.
        z=  fq.Fq(torch.tensor([8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861, 6631298214892334189, 1582556514881692819],dtype=torch.BLS12_381_Fq_G1_Mont))
        return ProjectivePointG1(x,y,z)
    else:
        # Z1Z1 = Z1^2
        z1z1 = F.mul_mod(self.z.value, self.z.value)

        # U2 = X2*Z1Z1
        u2 = F.mul_mod(other.x.value, z1z1)

        # S2 = Y2*Z1*Z1Z1
        s2 = F.mul_mod(other.y.value, self.z.value)
        s2 = F.mul_mod(s2, z1z1)

        if torch.equal(self.x.value, u2) and torch.equal(self.y.value, s2):
            # The two points are equal, so we double.
            return double_ProjectivePointG1(self)
        else:
            # H = U2-X1
            h = F.sub_mod(u2, self.x.value)

            # I = 4*(H^2)
            i = F.mul_mod(h, h)
            i = F.add_mod(i, i)
            i = F.add_mod(i, i)

            # J = H*I
            j = F.mul_mod(h, i)

            # r = 2*(S2-Y1)
            r = F.sub_mod(s2, self.y.value)
            r = F.add_mod(r, r)

            # V = X1*I
            v = F.mul_mod(self.x.value, i)

            # X3 = r^2 - J - 2*V
            x = F.mul_mod(r, r)
            x = F.sub_mod(x, j)
            v2 = F.add_mod(v, v)
            x = F.sub_mod(x, v2)

            # Y3 = r*(V-X3) - 2*Y1*J
            y = F.sub_mod(v, x)
            y = F.mul_mod(r, y)
            s1j = F.mul_mod(self.y.value, j)
            s1j2 = F.add_mod(s1j, s1j)
            y = F.sub_mod(y, s1j2)

            # Z3 = (Z1+H)^2 - Z1Z1 - H^2
            z = F.add_mod(self.z.value, h)
            z = F.mul_mod(z, z)
            z = F.sub_mod(z, z1z1)
            hh = F.mul_mod(h, h)
            z = F.sub_mod(z, hh)

            return ProjectivePointG1(fq.Fq(x), fq.Fq(y), fq.Fq(z))