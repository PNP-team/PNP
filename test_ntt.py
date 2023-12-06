import torch
import random
import torch.nn.functional as F

random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]
x = torch.tensor(random_list, dtype=torch.BLS12_381_Fr_G1_Mont)
x_cpu = torch.tensor(random_list,dtype=torch.uint64)

x_gpu = x.to("cuda")

output_cpu = torch.intt_zkp_(x_cpu)
z1 = output_cpu.tolist()

output = torch.intt_zkp(x_gpu)
z2 = output.to("cpu")
z2 = z2.tolist()
print(random_list[:5])
print(z2[:5])

print(z1==z2)

