import torch
import torch.nn as nn
import torchviz
import torch.nn.functional as F
import copy

class mod_add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_1,in_2):
        ctx.save_for_backward(in_1,in_2)
        output = F.sub_mod(in_1,in_2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input =grad_output
        return grad_input
    
class mod_sub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_1,in_2):
        ctx.save_for_backward(in_1,in_2)
        output = F.add_mod(in_1,in_2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input =grad_output
        return grad_input

class to_mod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_1):
        ctx.save_for_backward(in_1)
        output = F.to_mod(in_1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input =grad_output
        return grad_input   

# 构建模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

    def forward(self, input1, input2):
        # 合并两个椭圆曲线张量的输出
        input1=to_mod.apply(input1)
        input2=to_mod.apply(input2)
        combined =mod_add.apply(input1,input2)
        c = combined.clone()
        res=mod_sub.apply(c,input2)
        parameters_dict = dict(model.named_parameters())
        global combined_dict
        combined_dict={}
        combined_dict.update(parameters_dict)  # 将模型参数添加到新字典中
        combined_dict['input1'] = input1  # 添加输入数据 input1
        combined_dict['input2'] = input2  # 添加输入数据 input2
        combined_dict['c']=c
        return res

# 创建两个张量，设置requires_grad=False
# input1 = torch.tensor([1.0, 2.0], requires_grad=False,dtype=torch.float)
# input2 = torch.tensor([3.0, 4.0], requires_grad=False,dtype=torch.float)

input1 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base,requires_grad=True)
input2 =torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base,requires_grad=False)

# 创建自定义模型
model = CustomModel()

# 使用这些张量进行模型前向传播
output = model(input1, input2)

# 可视化模型结构
#torchviz.make_dot(output, params=dict(model.named_parameters())).render("test/11_11/pic/elliptic_curve_model_with_tensors", format="png")


# 创建一个新的字典，包含模型参数和输入数据


torchviz.make_dot(output, params=combined_dict,show_attrs=True, show_saved=True).render("test/11_11/pic/elliptic_curve_model_with_tensors", format="png")