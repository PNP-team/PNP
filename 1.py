import torch

a=torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]],dtype=torch.BLS12_381_Fr_G1_Mont)
a=a.tolist()
a+=a[:1]
print(a)
##实现
a=torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]],dtype=torch.BLS12_381_Fr_G1_Mont)
expanded_tensor = torch.concate((a, a[:1]), dim=0)