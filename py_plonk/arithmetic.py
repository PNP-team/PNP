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

def poly_add_poly(self: torch.Tensor, other: torch.Tensor):  
    if self.size(0) == 0:
        return other
    if other.size(0) == 0:
        return self
    N = min(len(self), len(other))
    return F.add_mod(self[:N], other[:N])

def MSM(bases,scalar):
    min_size = min(bases.size(0)//2, scalar.size(0))
    if min_size == 0:  #empty msm return zero_point
        commitment = ProjectivePointG1(fq.one(), fq.one(), fq.zero())
        return commitment
    else:
        commitment = F.multi_scalar_mult(bases.to("cuda"), scalar.to("cuda"))
        commitment = ProjectivePointG1(commitment[0],commitment[1],commitment[2])
        return commitment