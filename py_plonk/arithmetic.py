from .bls12_381 import fr,fq
from .jacobian import ProjectivePointG1
import random
import torch
import torch.nn.functional as F
import time
import torch

def calculate_execution_time(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()  # 记录函数开始执行的时间
            result = func(*args, **kwargs)
            end_time = time.time()    # 记录函数执行结束的时间
            execution_time = end_time - start_time  # 计算函数执行时间
            print(f"func {func.__name__} consumed: {execution_time} s")
            return result
        return wrapper

def tensor_to_int(input:torch.Tensor):
    list = input.tolist()
    output = 0 
    for i in reversed(list):
        output = output << 64
        output = output | i
    return output

def convert_to_bigints(p: torch.Tensor):
    if p.size(0) == 0:
        return F.to_base(p)
    else:
        res = F.to_base(p)
        return res

def skip_leading_zeros_and_convert_to_bigints(p: torch.Tensor):
    return convert_to_bigints(p.to('cpu'))


def poly_add_poly(self: torch.Tensor, other: torch.Tensor):    
    if self.size(0) == 0:
        res = other[:]
        return res
    if other.size(0) == 0:
        res =self[:]
        return res
    elif len(self) >= len(other):
        result = self.clone()
        F.add_mod(result[:len(other)],other,True)
        return result.to("cuda")
    else:
        result = other.clone()
        F.add_mod(result[:len(self)],self,True)
        return result.to("cuda")

def poly_mul_const(poly:torch.Tensor, elem:torch.Tensor):  
    if poly.size(0) == 0 :
        return poly
    else:
        result = F.mul_mod_scalar(poly,elem)
        return result


def rand_poly(d):
    random.seed(42)
    random_coeffs = [fr.make_tensor(random.random) for _ in range(d + 1)]
    return random_coeffs


def poly_add_poly_mul_const(self:torch.Tensor, f: torch.Tensor, other: torch.Tensor):
    if self.size(0) == 0 and other.size(0) == 0:
        return torch.tensor([], dtype = self.dtype, device = self.device)

    if self.size(0) == 0:
        other = other.to('cuda')
        self = other.clone()
        self = F.mul_mod_scalar(self, f)
        return self
    elif other.size(0) == 0:
        self = self.to('cuda')
        return self
    elif self.size(0) >= other.size(0):
        self = self.to('cuda')
        other = other.to('cuda')
    else:
        self = self.to('cuda')
        other = other.to('cuda')
        self = F.pad_poly(self, other.size(0))
    temp = F.mul_mod_scalar(other, f)
    F.add_mod(self[:other.size(0)], temp, True)
    
    return self

# The first lagrange polynomial has the expression:
# L_0(X) = mul_from_1_to_(n-1) [(X - omega^i) / (1 - omega^i)]
#
# with `omega` being the generator of the domain (the `n`th root of unity).
#
# We use two equalities:
#   1. `mul_from_2_to_(n-1) [1 / (1 - omega^i)] = 1 / n`
#   2. `mul_from_2_to_(n-1) [(X - omega^i)] = (X^n - 1) / (X - 1)`
# to obtain the expression:
# L_0(X) = (X^n - 1) / n * (X - 1)
def compute_first_lagrange_evaluation(size, z_h_eval, z_challenge):
    # single scalar OP on CPU
    one = fr.one()
    n_fr = fr.make_tensor(size)
    z_challenge_sub_one = F.sub_mod(z_challenge, one)
    denom = F.mul_mod(n_fr, z_challenge_sub_one)
    denom_in = F.div_mod(one, denom)
    res = F.mul_mod(z_h_eval, denom_in)
    return res  


def MSM_new(bases,scalar): #bases POINT scalar SCALAR
    min_size = min(bases.size(0)//2, scalar.size(0))
    
    if min_size == 0:  #empty msm return zero_point
        commitment = ProjectivePointG1(fq.one(), fq.one(), fq.zero())
        return commitment
    else:
        base = bases.clone()
        base = base[:min_size].view(-1, 6) # dim2 to 1
        base = base.to('cuda')
        scalar = scalar.to('cuda')
        commitment = F.multi_scalar_mult(base, scalar)
        commitment = ProjectivePointG1(commitment[0],commitment[1],commitment[2])
        return commitment