# Linear combination of a series of values
# For values [v_0, v_1,... v_k] returns:
# v_0 + challenge * v_1 + ... + challenge^k  * v_k
import torch.nn.functional as F
import torch
def extend_tensor(input:torch.tensor,size):
    res= torch.zeros(size,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    for i in res :
        i= input
    return res 
def Multiset_lc(values, challenge):
    kth_val = values.elements[-1]
    reverse_val=reversed(values.elements[:-1])
    # for val in reverse_val:
    #     for i in range(len(kth_val)):
    #         # kth_val[i] = kth_val[i].mul(challenge)
    #         kth_val[i]=F.mul_mod(kth_val[i],challenge)
    #         kth_val[i]=F.add_mod(kth_val[i],val[i])
    challenge= extend_tensor(challenge,len(kth_val))
    for val in reverse_val :
        kth_val=F.mul_mod(kth_val,challenge)
        kth_val=F.add_mod(kth_val,val)
    return kth_val

def lc(values:list, challenge):
    kth_val = values[-1]
    for val in reversed(values[:-1]):
        kth_val = F.mul_mod(kth_val, challenge)
        kth_val = F.add_mod(kth_val, val)

    return kth_val

