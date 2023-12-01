import torch
import torch.nn.functional as F
import torchviz
class MyNumber:   ######该类为了实现tensor运算符的重载
    def __init__(self, value):
        if(value.dtype==torch.BLS12_377_Fr_G1_Base):
            self.base = value
            self.mont=self.to_mont()
            self.size = value.size()
        else:
            self.mont=value
            self.base=self.to_base()
            self.size =value.size()
    # 重载加法操作符 +

    def to_mont(self):
        self.mont=F.to_mont(self.base)
        return self.mont
    
    def to_base(self):
        return MyNumber(F.to_base(self.mont))

    def __add__(self, other):
        res=F.add_mont(self.mont,other.mont)
        return MyNumber(res)
    
    def __sub__(self, other):
        res=F.sub_mont(self.mont,other.mont)
        return MyNumber(res)

    def __mul__(self,other):
        res=F.mul_mont(self.mont,other.mont)
        return MyNumber(res)
    
    def __div__(self,other):
        res=F.div_mont(self.mont,other.mont)
        return MyNumber(res)
    
    # def __str__(self):
    #     return str(self.value)


class MyCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output
        return grad_input

class MyClass(MyNumber):##该类为了实现运算过程
    def __init__(self, num):
        
        self.num=num
        # self.custom_function = MyCustomFunction.apply
    
    def compute(self, other):
        # 在 MyClass 中调用 MyNumber 的实例，并使用重载的操作符
        #self.num = self.num + other.mont
        # x=MyCustomFunction(other)
        # res=self.num.mont-x
        res2=self.num+other ##res is a tensor_mont
        return res2


# 创建 MyNumber 和 MyClass 实例


x1 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base)
x2 =torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_377_Fr_G1_Base)

num1 = MyNumber(x1)
num2 = MyNumber(x2)
# my_class=MyClass(num1)
temp1=num1+num2
temp2=temp1*num2
temp2=temp2.base
print(temp2.size)
# # 在 MyClass 中调用 MyNumber 的实例，并使用重载的操作符
# add_result= my_class.compute(num2)

# print("Addition:", F.to_base(add_result.mont))  # 输出 Addition: 15
torchviz.make_dot(temp2).render("new_model", format="pdf")