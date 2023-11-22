import torch
import torch.nn.functional as F

print(torch.__path__)

x1 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base)

x2 =torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base)
# x.to("cuda")
print("===========")
y1 = F.to_mont(x1)
y2=F.to_mont(x2)


# z = F.to_base(y)
# print(z)

# res=F.sub_mont(y1,y2)
# res=F.add_mont(res,y1)
res=F.sub_mont(y1,y2)
res2=F.add_mont(res,y1)
# a = y.clone()

print("mont",res2)
z=F.to_base(res2)
print("base",z)



# test cuda
x3 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8]], dtype=torch.BLS12_377_Fr_G1_Base)

x4 =torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_377_Fr_G1_Base)

y3 = F.to_mont(x3)
y4=F.to_mont(x4)
y3 = y3.cuda()
y4=y4.cuda()
#m = F.add_mont(y3,y4)
m=F.div_mont(y3,y4)

# y=y3/y4

m2=F.mul_mont(m,y4)
res=F.to_base(m)
print(res)





# y = torch.tensor([[9223372036854772, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]], dtype=torch.big_integer)


# y = torch.tensor([
#     [[922337203685477, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]],
#     [[922337203685477, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]]
# ], dtype=torch.big_integer)

# print(x)
# print(y)

# print(type(x.type()))

# # x.to_Fq(torch.uint192)


# # x.to("CUDA")

# print(x.shape)
# print(y.shape)

# print("===========")
# print(x)
# y = F.to_mont(x)
# print(y)

# # # x = torch.tensor([[9223372036854775809, 2, 3], [4, 5, 6]], dtype=torch.uint64)
# # # y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint64)
# # # z = x

# # # x = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.float8_e5m2)
# y = torch.tensor([[922337203685477, 2, 3, 10], [4, 5, 6, 8]], dtype=torch.big_integer)
# # x = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.field64)

# print(y)

# # z = torch.add(x, y, alpha=2)

# # print(z) 

# # m = torch.nn.MyFUNC(inplace=True)



# # x = m(x)
# # print(x.cpu())


# # x = x.cuda()
# # x = m(x) #now in GPU
# # print(x.cpu())

# # c = x.tolist()

# # d = torch.tensor(c, dtype=torch.uint64)

# # print(d)

import torch
import torchviz

class MyCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input * 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * 2
        return grad_input

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # In the model, use the custom function directly, not with apply
        self.custom_function = MyCustomFunction.apply

    def forward(self, x,x2):
        # In the forward pass, use the custom function directly
        x = self.custom_function(x)
        result=x+x2
        return result

# Create model instance
model = MyModel()

# Create input tensor
x = torch.tensor([3.0], requires_grad=True)
x2 = torch.tensor([3.0], requires_grad=True)
# Forward pass
res = model(x,x2)
print(res.size())
#参数手动添加  [('x',x)]
# Use torchviz to visualize the computation graph
torchviz.make_dot(res, params=dict([('x',x)]+[('x2',x2)]+[('res',res)])).render("my_model", format="pdf")