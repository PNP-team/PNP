import torch
import time 
a = torch.tensor(
        [
            8589934590,
            6378425256633387010,
            11064306276430008309,
            1739710354780652911,
        ],
        dtype=torch.BLS12_381_Fr_G1_Mont,
    ).to("cuda")

start1 = time.time()
b = a.repeat(1<<20,1)
elapse1 = time.time() - start1

c = torch.repeat_to_poly(a, 1<<20)
start2 = time.time()
elapse2 = time.time() -start2

print(elapse1)
print(elapse2)
print(b.data == c.data)