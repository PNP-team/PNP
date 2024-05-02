
import torch
import torch.nn.functional as F
R_inv=torch.tensor([1438719116766986304, 12353315018595135583, 8215699910850717290, 1999183251322740055],dtype=torch.BLS12_381_Fr_G1_Mont)

def into_repr(self):

    if torch.equal(self,torch.zeros(1,4,dtype=torch.BLS12_381_Fr_G1_Mont)):
        return self
    else:
        res=F.to_base(self)
        return res

def into_repr_fq(self):
    if torch.equal(self,torch.zeros(1,6,dtype=torch.BLS12_381_Fq_G1_Mont)):
        return self
    else:
        res=F.to_base(self)
        return res