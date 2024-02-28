import torch
import torch.nn as nn
import torch.nn.functional as F
import random

########### Check whether the currently used pytorch is local ###########
def check_torch():
    print(torch.__path__)

########### Check the correctness of to_mont() and to_base ###########
def check_mont():
    x = torch.tensor([[9223372036854772, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.PALLAS_Fq_G1_Base)
    print("===========")
    print(x)
    y = F.to_mont(x)
    print(y)
    z = F.to_base(y)
    print(z)
    print(x==z)


########### Check the correctness of ntt and intt with gpu ###########
def check_ntt():
    domain_size = 32
    nttclass = nn.Ntt(domain_size, torch.BLS12_377_Fr_G1_Mont)
    inttclass = nn.Intt(domain_size, torch.BLS12_377_Fr_G1_Mont)

    random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]

    x = torch.tensor(random_list, dtype=torch.BLS12_377_Fr_G1_Base)
    x_mont = F.to_mont(x)
    print(x_mont[:5])

    x_gpu = x_mont.to("cuda")
    x_ntt = nttclass.forward(x_gpu)
    ntt_out = x_ntt.to("cpu")
    ntt_out= ntt_out.tolist()
    print(ntt_out[:5])

    x_intt = inttclass.forward(x_ntt)
    intt_out = x_intt.to("cpu")
    intt_out= intt_out.tolist()
    print(intt_out[:5])
    print(intt_out == x_mont.tolist())

########### Check the correctness of ntt and intt on coset with gpu ###########
def check_ntt_coset():
    domain_size = 32
    nttclass = nn.Ntt_coset(domain_size, torch.BLS12_377_Fr_G1_Mont)
    inttclass = nn.Intt_coset(domain_size, torch.BLS12_377_Fr_G1_Mont)

    random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]

    x = torch.tensor(random_list, dtype=torch.BLS12_377_Fr_G1_Base)
    x_mont = F.to_mont(x)
    print(x_mont[:5])

    x_gpu = x_mont.to("cuda")
    x_ntt = nttclass.forward(x_gpu)
    ntt_out = x_ntt.to("cpu")
    ntt_out= ntt_out.tolist()
    print(ntt_out[:5])

    x_intt = inttclass.forward(x_ntt)
    intt_out = x_intt.to("cpu")
    intt_out= intt_out.tolist()
    print(intt_out[:5])
    print(intt_out == x_mont.tolist())