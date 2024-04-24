import torch
import torch.nn.functional as F

a = torch.tensor([1, 2, 3], dtype=torch.FHE_PRIME_0, device='cuda:0')
b = torch.tensor([4, 5, 6], dtype=torch.FHE_PRIME_0, device='cuda:0')

a = F.to_mont(a)
print(a)
a = F.to_base(a)
print(a)
print(b)

c = F.add_mod(a, b)

print(c)