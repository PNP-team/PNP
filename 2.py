import torch
import torch.nn as nn
import torchviz
import torch.nn.functional as F


# 自定义椭圆曲线张量
class EllipticCurveTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # 这里进行椭圆曲线张量的操作
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 椭圆曲线张量的梯度计算规则
        grad_input = None
        return grad_input

# 构建模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # self.fc1 = nn.Linear(2, 1)

    def forward(self, input1, input2):
        # 使用自定义椭圆曲线张量进行前向传播
        output1 = EllipticCurveTensor.apply(input1)
        output2 = EllipticCurveTensor.apply(input2)
        
        # 合并两个椭圆曲线张量的输出
        input1=F.to_mont(input1)
        input2=F.to_mont(input2)
        combined =F.add_mont(input1,input2)
        # output=self.fc1(combined)
        return combined

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
torchviz.make_dot(output, params=dict(model.named_parameters())).render("test/11_11/pic/elliptic_curve_model_with_tensors", format="png")
