import gmpy2
import itertools

import torch
import time
import numpy as np
import torch.nn as nn
import copy
from .arithmetic import INTT
def resize_1(self, target_len):
    # res = copy.deepcopy(self)
    # self=from_tensor_list(self)
    # from_list_gmpy(self)
    if isinstance(self, list):
        res = copy.deepcopy(self)
    else:
        res = self.clone()
    
    if len(res)==0 :
        return  torch.zeros(target_len,4,dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    if len(self) < target_len:
        
        output=torch.zeros(target_len,4,dtype=torch.BLS12_381_Fr_G1_Mont)
        res = res.to('cpu')
        output[:len(self)]=res
        return output.to('cuda') 
    else :
        return res

def INTT_new(size,evals:torch.Tensor):

    inttclass = nn.Intt(size, torch.BLS12_381_Fr_G1_Mont)
    evals_resize=resize_1(evals,size)
    if evals_resize.device =='cuda':
        pass
    else:
       evals_resize=evals_resize.to('cuda')
    res= inttclass.forward(evals_resize)
    return res


def test():
    date_set2=["../../data/pp-3.npz","../../data/pk-3.npz","../../data/cs-3.npz","../../data/w_l_scalar_scalar-3.npy","../../data/w_r_scalar_scalar-3.npy","../../data/w_o_scalar_scalar-3.npy","../../data/w_4_scalar_scalar-3.npy"]
    date_set2=["../../data/pp-17.npz","../../data/pk-17.npz","../../data/cs-17.npz","../../data/w_l_scalar-17.npy","../../data/w_r_scalar-17.npy","../../data/w_o_scalar-17.npy","../../data/w_4_scalar-17.npy"]

    w_l_scalar=torch.tensor(np.load(date_set2[3],allow_pickle=True),dtype=torch.BLS12_381_Fr_G1_Mont)
    w_l_scalar=w_l_scalar.to('cuda')

    st=time.time()
    for i in range(100):
        w_l_scalar_intt=INTT(1024,w_l_scalar)
    ed=time.time()
    sum=ed-st
    print(sum)
