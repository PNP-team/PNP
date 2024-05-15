import torch


def to_tensor(input):
    output = []
    for j in range(4):
        output.append( int(input & 0xFFFFFFFFFFFFFFFF) )
        input = input >> 64
    output=torch.tensor(output,dtype=torch.BLS12_381_Fr_G1_Mont)
    return output

input = 13262374693698910701929044844600465831413122818447359594527400194675274060458
print(to_tensor(input))