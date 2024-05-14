import torch


def to_tensor(input):
    output = []
    for j in range(6):
        output.append( int(input & 0xFFFFFFFFFFFFFFFF) )
        input = input >> 64
    output=torch.tensor(output,dtype=torch.BLS12_381_Fq_G1_Mont)
    return output

input = 16009638220886669555301211019743182050671919363793336627014255617001491590008206123606004126824334192306356085582506
print(to_tensor(input))