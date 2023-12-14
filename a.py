import torch
import torch.nn.functional as F
import unittest
import math
import random

from sympy import mod_inverse

print(torch.__path__)

mod1=0xffffffff00000001
mod2=0x53bda402fffe5bfe
mod3=0x3339d80809a1d805
mod4=0x73eda753299d7d48

mod = mod1 + (mod2 << 64) + (mod3 << 128) + (mod4 << 192)
# print(mod)
t1=torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)

t2=torch.tensor([[2, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)

t3=torch.tensor([[1,0,0,0]],dtype=torch.BLS12_381_Fr_G1_Base)
t_mont = F.to_mont(t1)
print(t_mont)

r1=torch.tensor([[ 7329914482735656649, 17609816913662305860,  2036032305164203203,
          5556226402374631496],
        [ 3294422756502364090,  614120333364951411,  9150370761219685937,
          1747744114183546630]],dtype=torch.BLS12_381_Fr_G1_Mont)


r2=torch.tensor([[ 7329914491325591239,  5541498096586141254, 13100338581594211513,
          7295936757155284407],
        [ 3294422756502364090,   614120333364951411,  9150370761219685937,
          1747744114183546630]],dtype=torch.BLS12_381_Fr_G1_Mont)

r3=torch.tensor([[          8589934590,  6378425256633387010, 11064306276430008309,
          1739710354780652911]], dtype=torch.BLS12_381_Fr_G1_Mont)
def compute_base(in_a):
    in_a=in_a.tolist()
    rows, cols =len(in_a), len(in_a[0])
    for i in range(1):
        res=0
        for j in range(cols):
            res+=(int(in_a[i][j]))*(2**(j*64))%mod
            # if(j==3):
        res=(res*(2**256))%mod
    return res

def compute_base_1(in_a):
    in_a=in_a.tolist()
    rows, cols =len(in_a), len(in_a[0])
    for i in range(1):
        res=0
        for j in range(cols):
            res+=(int(in_a[i][j]))*(2**(j*64))
            # if(j==3):
    return res


def compute_mont(in_a):
    in_a=in_a.tolist()
    rows, cols =len(in_a), len(in_a[0])
    for i in range(1):
        res=0
        for j in range(cols):
            res+=(int(in_a[i][j]))*(2**(j*64))
    return res

base1=compute_base(t1)

base2=compute_base(t2)

mont1=compute_mont(r1)

mont2=compute_mont(r2)

mont3=compute_mont(r3)

# print((mont1*mont2)%mod)
# # print(mod_inverse(2**256,mod)*2**256%mod)
# # print(base1+base2)                                                                      

# print(compute_mont(F.add_mod(r1,r2)))
# print(((compute_mont(F.mul_mod(r1,r2)))<<256) %mod)



def test_add():


    min_dimension = 4 

    rows = random.randint(min_dimension, min_dimension) 
    columns = random.randint(min_dimension, min_dimension)  

  
    random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
    random_array_2 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
    t1 = torch.tensor(random_array_1,dtype=torch.BLS12_381_Fr_G1_Base)
    t2 = torch.tensor(random_array_2,dtype=torch.BLS12_381_Fr_G1_Base)
    #################################################################
    base1=compute_base(t1)
    base2=compute_base(t2)

    mont1=compute_mont(r1)
    mont2=compute_mont(r2)
    # print((mont1*mont2)%mod)
    # print(compute_mont(F.mul_mod(r1,r2)))
    # if((mont1*mont2)%mod==compute_mont(F.add_mod(r1,r2))):
    #     print("sucess")


test_add()

# t_res=F.to_mont(t)
# print(t_res)




