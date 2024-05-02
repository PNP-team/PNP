import torch
a=torch.cuda.device_count()
print(a)
a=torch.zeros(1,4)
a=a.to('cuda:1')
print(a.device)