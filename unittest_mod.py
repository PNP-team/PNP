import torch
import torch.nn.functional as F
import os
import unittest

from HTMLTestRunner import HTMLTestRunner
print(torch.__path__)

x1 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base)

x2 =torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base)
# x.to("cuda")
print("===========")
y1 = F.to_mod(x1)
y2=F.to_mod(x2)

# res=F.sub_mont(y1,y2)
# res=F.add_mont(res,y1)
res=F.sub_mod(y1,y2)
res2=F.add_mod(res,y2)
# a = y.clone()

print("mont",res2)
z=F.to_base(res2)
print("base",z)

import torch
import torchviz

class CustomTestCase(unittest.TestCase):
    def assertCustomEqual(self, first, second, msg=None):
        self.assertEqual(first, second, msg=msg)

    # 在测试方法中使用新的自定义断言方法
    def test_custom_data_equality(self):
        custom_data_1 =y1
        custom_data_2 =y2

        # 使用自定义断言方法进行比较
        self.assertCustomEqual(custom_data_1, custom_data_2, "自定义数据应该相等")
    
    def test_custom_data_type(self):
        custom_data_1=y1
        self.assertIsInstance(x1, torch.BLS12_377_Fr_G1_Base)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CustomTestCase))
    with open('HTMLReport.html', 'w') as f:
        runner = HTMLTestRunner(stream=f,
                                title='MathFunc Test Report',
                                description='generated by HTMLTestRunner.',
                                verbosity=2
                                )
        runner.run(suite)

# class MyCustomFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         output = input * 2
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output * 2
#         return grad_input



# class MyModel(torch.nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # In the model, use the custom function directly, not with apply
#         self.custom_function = MyCustomFunction.apply

#     def forward(self, x,x2):
#         # In the forward pass, use the custom function directly
#         x = self.custom_function(x)
#         result=x+x2
#         return result

# # Create model instance
# model = MyModel()

# # Create input tensor
# x1 = torch.tensor([3.0], requires_grad=False)
# x2 = torch.tensor([3.0], requires_grad=False)
# # Forward pass
# res = model(x1,x2)
# print(res.size())
# #参数手动添加  [('x',x)]
# # Use torchviz to visualize the computation graph

# torchviz.make_dot(res, params=dict([('x1',x1)]+[('x2',x2)]+[('res',res)])).render("test/11_11/pic/my_model1", format="pdf")