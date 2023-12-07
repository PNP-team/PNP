import torch
import torch.nn.functional as F
import unittest



print(torch.__path__)

# x = torch.tensor([[9223372036854772, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)
# xq = torch.tensor([[9223372036854772, 2, 3, 4, 5, 6], [1, 2, 5, 6, 7, 8]], dtype=torch.BLS12_381_Fq_G1_Base)
# # x.to("cuda")
# print("===========")
# print(x)
# y = F.to_mod(x)
# print(y)
# z = F.to_base(y)
# print(z)

# a = y.clone()
# print(a)

mod1=0xffffffff00000001
mod2=0x53bda402fffe5bfe
mod3=0x3339d80809a1d805
mod4=0x73eda753299d7d48

# mod1 = 0x0a1180000000000
# mod2 = 0x59aa76fed0000001
# mod3=0x60b44d1e5c37b001
# mod4=0x12ab655e9a2ca556


mod = mod1 + (mod2 << 64) + (mod3 << 128) + (mod4 << 192)
print(mod)
t=torch.tensor([[1, 0,0,0],[1,0,0,0]], dtype=torch.BLS12_381_Fr_G1_Base)
t_mont = F.to_mod(t)
print(F.to_base(t_mont))

print(hex(8589934590))
print(hex(6378425256633387010))
print(hex(11064306276430008309))
print(hex(1739710354780652911))

t1=torch.tensor([[1, 0,0,0],[1,0,0,0]], dtype=torch.uint64)
r1=torch.tensor([[          8589934590,  6378425256633387010, 11064306276430008309,
          1739710354780652911],
        [          8589934590,  6378425256633387010, 11064306276430008309,
          1739710354780652911]], dtype=torch.uint64)

#r1 = r1.tolist()

print(r1)
def compute_base(in_a):
    rows, cols = in_a.shape
    for i in range(1):
        res=0
        for j in range(cols):
            res+=(int(in_a[i][j]))*(2**(j*64))%mod
            # if(j==3):
        res=(res*(2**256))%mod
    print(res)
    
def compute_mont(in_a):
    rows, cols = in_a.shape
    for i in range(1):
        res=0
        for j in range(cols):
            print(in_a[i][j])
            res+=(int(in_a[i][j]))*(2**(j*64))
    print(res)

compute_base(t1)
compute_mont(r1)

t_res=F.to_mod(t)
print(t_res)
# y = torch.tensor([[9223372036854772, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]], dtype=torch.big_integer)


# y = torch.tensor([
#     [[922337203685477, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]],
#     [[922337203685477, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]]
# ], dtype=torch.big_integer)

# print(x)
# print(y)

# print(type(x.type()))

# # x.to_Fq(torch.uint192)
print(11<<1)

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


