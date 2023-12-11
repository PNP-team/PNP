import torch
import torch.nn.functional as F
import unittest

import random

print(torch.__path__)

mod1=0xffffffff00000001
mod2=0x53bda402fffe5bfe
mod3=0x3339d80809a1d805
mod4=0x73eda753299d7d48

mod = mod1 + (mod2 << 64) + (mod3 << 128) + (mod4 << 192)
# print(mod)
t1=torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)

t2=torch.tensor([[2, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)

t_mont = F.to_mont(t2)
# print(t_mont)

r1=torch.tensor([[7329914504210493124, 5885763944681445961, 2026681885674896553,
          682130252471487967],
        [ 3294422756502364090,  614120333364951411,  9150370761219685937,
          1747744114183546630]],dtype=torch.BLS12_381_Fr_G1_Mont)


r2=torch.tensor([[ 7329914491325591239,  5541498096586141254, 13100338581594211513,
          7295936757155284407],
        [ 3294422756502364090,   614120333364951411,  9150370761219685937,
          1747744114183546630]],dtype=torch.BLS12_381_Fr_G1_Mont)

def compute_base(in_a):
    in_a=in_a.tolist()
    rows, cols =len(in_a), len(in_a[0])
    for i in range(1):
        res=0
        for j in range(cols):
            res+=(int(in_a[i][j]))*(2**(j*64))%mod
            # if(j==3):
        res=(res*(2**256))%mod
    return res
    
def compute_mont(in_a):
    in_a=in_a.tolist()
    rows, cols =len(in_a), len(in_a[0])
    for i in range(1):
        res=0
        for j in range(cols):
            res+=(int(in_a[i][j]))*(2**(j*64))
    return res

base1=compute_base(t1)
base2=compute_base(t2)

mont1=compute_mont(r1)
mont2=compute_mont(r2)

# print(mont1,mont2)
print( (mont1*mont2)%mod)
print(compute_mont(F.add_mod(r1,r2)))
print(compute_mont(F.mul_mod(r1,r2)))



def test_add():

    #####生成测试数据
    min_dimension = 4  # 最低维度为4

    # 随机确定行数和列数（至少4行4列）
    rows = random.randint(min_dimension, min_dimension)  # 至少4行，最多7行
    columns = random.randint(min_dimension, min_dimension)  # 至少4列，最多7列

    # 生成随机整数二维数组
    random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
    random_array_2 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
    t1 = torch.tensor(random_array_1,dtype=torch.BLS12_381_Fr_G1_Base)
    t2 = torch.tensor(random_array_2,dtype=torch.BLS12_381_Fr_G1_Base)
    #################################################################
    base1=compute_base(t1)
    base2=compute_base(t2)

    mont1=compute_mont(r1)
    mont2=compute_mont(r2)
    # print((mont1*mont2)%mod)
    # print(compute_mont(F.mul_mod(r1,r2)))
    # if((mont1*mont2)%mod==compute_mont(F.add_mod(r1,r2))):
    #     print("sucess")


test_add()

# t_res=F.to_mont(t)
# print(t_res)


# import unittest
# import torch
# import pytest

# class TestTensorTypes(unittest.TestCase):


#     @pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float64])  # 在这里添加更多的数据类型
#     def test_tensor_dtype(dtype):
#         def my_function(in_a):
#             if in_a.dtype == torch.float64:
#                 return False
#             else:
#                 return True
#         tensor = torch.zeros(5, dtype=dtype)  # 创建不同 dtype 的张量
#         result = my_function(tensor)  # 调用类方法需要使用 self
#         assert result is True, f"Failed for dtype: {dtype}"  # 使用 assert 进行断言验证

# def run_tests():
#     test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTensorTypes)
#     unittest.TextTestRunner().run(test_suite)

#     # 添加更多的测试方法
# run_tests()
# # 你的其他测试类和函数


