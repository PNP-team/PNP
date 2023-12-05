import torch
import torch.nn.functional as F
import unittest



print(torch.__path__)

x1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_377_Fr_G1_Base)
x2 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_377_Fr_G1_Base)
# x.to("cuda")
print("===========")
print(x1)
y1 = F.to_mod(x1)

y2= F.to_mod(x2)

z = F.add_mod(y1,y2)

print(F.to_base(z))
if(torch.equal_mod(y1,y1)):
    print(1)

# if(y1==F.to_mod(x1)):
#     print(1)
# class TestDict(unittest.TestCase):

#     def test_init(self):
#         d =F.to_mod(x1)
#         self.assertEqual(d,y1)



# if __name__ == '__main__':
#     unittest.main()


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



import unittest
import torch
import pytest

class TestTensorTypes(unittest.TestCase):


    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.float64])  # 在这里添加更多的数据类型
    def test_tensor_dtype(dtype):
        def my_function(in_a):
            if in_a.dtype == torch.float64:
                return False
            else:
                return True
        tensor = torch.zeros(5, dtype=dtype)  # 创建不同 dtype 的张量
        result = my_function(tensor)  # 调用类方法需要使用 self
        assert result is True, f"Failed for dtype: {dtype}"  # 使用 assert 进行断言验证

def run_tests():
    test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTensorTypes)
    unittest.TextTestRunner().run(test_suite)

    # 添加更多的测试方法
run_tests()
# 你的其他测试类和函数


