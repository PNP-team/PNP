import torch
import random
import sys
import torch.nn as nn
import torch.nn.functional as F
from load import read_pp_data,read_scalar_data,from_gmpy_tensor,from_list_gmpy
# xr = torch.tensor([[random.randint(1, 1000000) for _ in range(4)] for _ in range(1024)], dtype = torch.BLS12_381_Fr_G1_Base)
# xq = torch.tensor([[random.randint(1, 1000000) for _ in range(6)] for _ in range(1024)], dtype = torch.BLS12_381_Fq_G1_Base)


# inputr = xr.to("cuda")
# inputq = xq.to("cuda")

# montr = F.to_mont(inputr)
# montq = F.to_mont(inputq)

MODULUS= 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787

R=3380320199399472671518931668520476396067793891014375699959770179129436917079669831430077592723774664465579537268733

R2=2708263910654730174793787626328176511836455197166317677006154293982164122222515399004018013397331347120527951271750

R_INV=3231460744492646417066832100176244795738767926513225105051837195607029917124509527734802654356338138714468589979680

MODULUS_r = 52435875175126190479447740508185965837690552500527637822603658699938581184513

R_r = 10920338887063814464675503992315976177888879664585288394250266608035967270910

R_INV_r = 12549076656233958353659347336803947287922716146853412054870763148006372261952


def inverse(self):
    if self == 0:
            print("cannot invert 0!\n")
            return  None
    u = self
    one = 1
    v = MODULUS
    b = R2
    c = 0

    while u != one and v != one:
        while u & 1 == 0:
            u = u // 2
            if b & 1 == 0:
                b = b // 2
            else:
                b = b + MODULUS
                b = b // 2
        while v & 1 == 0:
            v =v // 2
            if c & 1 == 0:
                c = c // 2
            else:
                c = c + MODULUS
                c = c // 2
        if v < u:
            u = u-v
            if c > b:
                b = b + MODULUS
            b = b - c
            b = b%MODULUS
        else:
            v = v-u
            if b > c:
                c = c + MODULUS
            c = c - b
            c = c%MODULUS
    if u == one:
        return b
    else:
        return c
        
def into_repr(self):
    if self == 0:
        return self
    else:
        res = self * R_INV_r
        res %= MODULUS_r
        return res
    
def into_repr_q(self):
    if self == 0:
        return self
    else:
        res = self * R_INV
        res %= MODULUS
        return res
    
def convert_to_bigints(p):
    coeffs = [into_repr(s) for s in p]
    return coeffs

def convert_to_bigints_q(p):
    coeffs = [into_repr_q(s) for s in p]
    return coeffs

pp = read_pp_data("params.txt")
w_l_scalar = read_scalar_data("w_l_scalar.txt")
pp = pp[:2048]
# pp_gmpy = []
# for i in range(len(pp)):
#     mem = 0 
#     pp[i].reverse()
#     for j in pp[i]:
#         mem = mem<<64
#         mem = mem | j
#     pp_gmpy.append(mem)
# pp_gmpy = convert_to_bigints_q(pp_gmpy)
# pp_repr = from_gmpy_tensor(pp_gmpy,6,torch.BLS12_381_Fq_G1_Base)
# # 将数据写入文件
# with open('pp.txt', 'w') as file:
#     for sublist in pp_repr:
#         file.write(' '.join(map(str, sublist)) + '\n')  

# scalar_list = [[8589934590,6378425256633387010,11064306276430008309,1739710354780652911] for _ in range(1024)]
# x=[6679831729115696150,8653662730902241269,1535610680227111361,17342916647841752903,17135755455211762752,1297449291367578485]
# y=[13451288730302620273,10097742279870053774,15949884091978425806,5885175747529691540,1016841820992199104,845620083434234474]
# point_list=[]
# for i in range(1024):
#     point_list.append(x)
#     point_list.append(y)
# for i in range(5):  
#     point_list[2+i*2] = [0,0,0,0,0,0]
#     point_list[3+i*2] = [0,0,0,0,0,0]
#     #point_list[3+i*2] = [8505329371266088957,17002214543764226050,6865905132761471162,8632934651105793861,6631298214892334189,1582556514881692819]
# point_list[-1] = [0,0,0,0,0,0]
# point_list[-2] = [0,0,0,0,0,0]
# point_list[-3] = [0,0,0,0,0,0]
# point_list[-4] = [0,0,0,0,0,0]

# for i in range(10):
#     scalar_list[i] = [i+1,i+2,i+3,i+4]

    
scalar = torch.tensor(w_l_scalar, dtype = torch.BLS12_381_Fr_G1_Mont)
point = torch.tensor(pp, dtype = torch.BLS12_381_Fq_G1_Mont)
scalar_gpu = scalar.to("cuda")
point_gpu = point.to("cuda")

domain_size = 32
nttclass = nn.Intt(domain_size, torch.BLS12_381_Fr_G1_Mont)
coeff_gpu = nttclass.forward(scalar_gpu)
coeff_cpu = coeff_gpu.to("cpu")
coeff = coeff_cpu.tolist()
coeff_gmpy = []
for i in range(len(coeff)):
    mem = 0 
    coeff[i].reverse()
    for j in coeff[i]:
        mem = mem<<64
        mem = mem | j
    coeff_gmpy.append(mem)
coeff_gmpy = convert_to_bigints(coeff_gmpy)
msmscalar = from_gmpy_tensor(coeff_gmpy,4,torch.BLS12_381_Fr_G1_Base)

    
msmcoeff = msmscalar.to("cuda")

# Choose which CUDA device to use (e.g., device 0)
device = torch.device(f"cuda:{0}")
properties = torch.cuda.get_device_properties(device)
smcount = properties.multi_processor_count

step1 = torch.msm_zkp(point_gpu, msmcoeff, smcount, dtype=torch.BLS12_381_Fq_G1_Base, device="cuda")
step1res = step1.to("cpu")
list1=step1res.tolist()
step2res = torch.msm_collect(step1res,1024)
output = step2res.tolist()

#to_affine
if(sum(output[-1])==0):
    output = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
elif(output[-1]==[8505329371266088957,17002214543764226050,6865905132761471162,
                8632934651105793861,6631298214892334189,1582556514881692819]):
    output=[output[0],output[1]]
else:
    x=0
    y=0
    z=0
    for j in reversed(output[0]):
        x= x<<64
        x= x|j
    for j in reversed(output[1]):
        y= y<<64
        y= y|j
    for j in reversed(output[2]):
        z= z<<64
        z= z|j
    zinv = inverse(z)
    zinv_squred = (zinv*zinv*R_INV)%MODULUS
    x = (x*zinv_squred*R_INV)%MODULUS
    y = ((y*(zinv_squred*zinv*R_INV)%MODULUS)*R_INV)%MODULUS
    xx=[]
    yy=[]
    for j in range(6):
        xx.append(x&0xffffffffffffffff)
        x = x>>64
    for j in range(6):
        yy.append(y&0xffffffffffffffff)
        y = y>>64
    output = [xx,yy]
print(output)
