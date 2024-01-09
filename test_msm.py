import torch
import random
import torch.nn.functional as F

xr = torch.tensor([[random.randint(1, 1000000) for _ in range(4)] for _ in range(1024)], dtype=torch.BLS12_381_Fr_G1_Base)
xq = torch.tensor([[random.randint(1, 1000000) for _ in range(6)] for _ in range(1024)], dtype=torch.BLS12_381_Fq_G1_Base)


inputr = xr.to("cuda")
inputq = xq.to("cuda")

montr = F.to_mont(inputr)
montq = F.to_mont(inputq)

out = torch.msm_zkp(montq, montr)

print(1)