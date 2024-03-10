import torch
import random
import sys
import torch.nn as nn
import torch.nn.functional as F
# from load import read_pp_data,read_scalar_data,from_gmpy_tensor,from_list_gmpy
# xr = torch.tensor([[random.randint(1, 1000000) for _ in range(4)] for _ in range(1024)], dtype = torch.BLS12_381_Fr_G1_Base)
# xq = torch.tensor([[random.randint(1, 1000000) for _ in range(6)] for _ in range(1024)], dtype = torch.BLS12_381_Fq_G1_Base)


# inputr = xr.to("cuda")
# inputq = xq.to("cuda")

# montr = F.to_mont(inputr)
# montq = F.to_mont(inputq)

MODULUS = 52435875175126190479447740508185965837690552500527637822603658699938581184513

R_INV = 12549076656233958353659347336803947287922716146853412054870763148006372261952


def into_repr(self):
    if self == 0:
        return self
    else:
        res = self * R_INV
        res %= MODULUS
        return res
    
def convert_to_bigints(p):
    coeffs = [into_repr(s) for s in p]
    return coeffs

# pp = read_pp_data("params.txt")
# w_l_scalar = read_scalar_data("w_l_scalar.txt")
x = [6679831729115696150,8653662730902241269,1535610680227111361,17342916647841752903,17135755455211762752,1297449291367578485]
y = [13451288730302620273,10097742279870053774,15949884091978425806,5885175747529691540,1016841820992199104,845620083434234474]
point_list = []
for i in range(1024):
    point_list.append(x)
    point_list.append(y)

scalar = torch.tensor([[8589934590,6378425256633387010,11064306276430008309,1739710354780652911] for _ in range(1024)], dtype = torch.BLS12_381_Fr_G1_Mont)
# scalar = torch.tensor(w_l_scalar, dtype = torch.BLS12_381_Fr_G1_Mont)
# point = torch.tensor(pp, dtype = torch.BLS12_381_Fq_G1_Mont)
point = torch.tensor(point_list, dtype = torch.BLS12_381_Fq_G1_Mont)

# print(point[:4])
scalar_gpu = scalar.to("cuda")
point_gpu = point.to("cuda")
# domain_size = 32
# nttclass = nn.Ntt(domain_size, torch.BLS12_381_Fr_G1_Mont)
# coeff_gpu = nttclass.forward(scalar_gpu)
# coeff_cpu = coeff_gpu.to("cpu")
# coeff = coeff_cpu.tolist()
# coeff_gmpy = []
# for i in range(len(coeff)):
#     mem = 0 
#     coeff[i].reverse()
#     for j in coeff[i]:
#         mem = mem<<64
#         mem = mem | j
#     coeff_gmpy.append(mem)
# coeff_gmpy = convert_to_bigints(coeff_gmpy)
# print(coeff[:10])
# msmscalar = from_gmpy_tensor(coeff_gmpy,4,torch.BLS12_381_Fr_G1_Base)
# msmcoeff = msmscalar.to("cuda")

step1 = torch.msm_zkp(point_gpu, scalar_gpu)
step1res = step1.to("cpu")
print(step1res[:10])
#step2res = torch.msm_collect(step1res,1024)
# output = step2res.tolist()
# print(output)
# y=[1395070632402625355, 18168135245787491657, 719421310633343711, 9299717603087955920, 5549971297597933013, 114424826436245817]
# Y=0
# for i in y:
#     Y = Y<<64
#     Y = Y|i
# print(hex(Y))
# out = []
# for i in range(len(output)):
#     mem = 0 
#     for j in reversed(output[i]):
#         mem = mem << 64
#         mem = mem | j
#     out.append(mem)
# # out = from_list_gmpy(output)
# print(out)