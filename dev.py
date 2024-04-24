import torch
import torch.nn.functional as F

a = torch.tensor([1 for _ in range(256)], dtype=torch.FHE_PRIME_0, device='cuda:0')
b = torch.tensor([4 for _ in range(256)], dtype=torch.FHE_PRIME_0, device='cuda:0')

aa = F.to_mont(a)
bb = F.to_mont(b)

print(aa)

dd = F.div_mod(bb, aa)
ddd = F.to_base(dd)


print(ddd)