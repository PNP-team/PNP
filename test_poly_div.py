import torch
import torch.nn.functional as F
import json
import time

c = torch.tensor([15246957462666058365,  3560464253533287514,   499736670904067688,
         7202556651244979518], dtype=torch.BLS12_381_Fr_G1_Mont).to("cuda")
with open('divid.json', 'r') as f:
    input = json.load(f)
input = torch.tensor(input, dtype=torch.BLS12_381_Fr_G1_Mont).to("cuda")
start = time.time()
out = F.poly_div_poly(input, c)
elapse = time.time() - start
with open('quotient.json', 'r') as f:
    quotient = json.load(f)
out = out.tolist()
for i in(range(len(quotient))):
    if(quotient[i] != out[i]):
        print(i)
        break
print("hhh")
