import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__path__)

x = torch.tensor([[9223372036854772, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.PALLAS_Fq_G1_Base)
xq = torch.tensor([[9223372036854772, 2, 3, 4, 5, 6], [1, 2, 5, 6, 7, 8]], dtype=torch.BLS12_381_Fq_G1_Base)
# x.to("cuda")
print("===========")
print(x)
y = F.to_mont(x)
print(y)
z = F.to_base(y)
print(z)
