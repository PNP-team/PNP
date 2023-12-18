import torch
import random
import torch.nn as nn
import torch.nn.functional as F

WINDOW_SIZE = 1 << 14
WINDOW_NUM = 2
domain_size = 32
inttclass = nn.NTT_zkp(True, 0, domain_size, torch.BLS12_377_Fr_G1_Mont)

random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]
x = torch.tensor(random_list, dtype=torch.BLS12_377_Fr_G1_Mont)

x_gpu = x.to("cuda")

output = inttclass.forward(x_gpu, True, False)
z2 = output.to("cpu")
z2 = z2.tolist()
print(random_list[:5])
print(z2[:5])




