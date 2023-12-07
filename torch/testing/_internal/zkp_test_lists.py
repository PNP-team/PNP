import torch.nn.functional as F
import torch
import torch
import torch.nn.functional as F
import os
import unittest
import torchviz
'''
测试的主要内容：
1.测试mont的转换正确性 
2.测试add等运算符的运算的正确性
3.验证不同曲线的正确性
'''
data_type=[torch.BLS12_377_Fr_G1_Base,torch.BLS12_381_Fr_G1_Base]

class CustomTestCase(unittest.TestCase):

    def setUp(self):
        self.x1 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_381_Fr_G1_Base)
        self.x2 =torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_381_Fr_G1_Base)
        comput_mod_func = lambda mod1, mod2, mod3, mod4: mod1 + (mod2 << 64) + (mod3 << 128) + (mod4 << 256)
        self.modlist={
    torch.BLS12_377_Fr_G1_Base: comput_mod_func(0xffffffff00000001,0x53bda402fffe5bfe,0x3339d80809a1d805,0x73eda753299d7d48),
    torch.BLS12_381_Fr_G1_Base: comput_mod_func(0x0a11800000000001,0x59aa76fed0000001,0x60b44d1e5c37b001,0x12ab655e9a2ca556)
}

    def compute_base(self,in_a,mod):
            ##确保是uint64
            rows, cols = in_a.shape
            for i in range(1):
                res=0
                for j in range(cols):
                    res+=(int(in_a[i][j])<<256)*(2**(j*64))%mod
            return res
    
    def compute_mont(self,in_a,mod):
        rows, cols = in_a.shape
        for i in range(1):
            res=0   
            for j in range(cols):
                res+=(int(in_a[i][j]))*(2**(j*64))%mod
        return res

    def assertCustomEqual(self, first, second, msg=None):
         if not torch.equal_mod(first, second):
            raise ValueError("Tensors are not equal")
        # self.assertEqual(first, second, msg=msg)



    def test_custom_data_equality(self):
        #蒙哥马利域下的比较
        custom_data_1 =F.to_mod(self.x1)
        custom_data_2 =F.to_mod(self.x2)
        # print(custom_data_1,custom_data_2)
        self.assertCustomEqual(custom_data_1, custom_data_2, "自定义数据应该相等")
    
    def test_custom_data_type(self):
        #检查mont转换的结果是否正确，和直接计算的结果比对
        def simple_type_test(in_a):
            try:
                custom_data_1=F.to_mod(self.x1)
                test_to_base_value=F.to_base(custom_data_1)
                self.assertEqual(test_to_base_value.dtype, torch.BLS12_381_Fr_G1_Base )
            except Exception as e:
                print(f"An error occurred: {e}")
                return False

        for type_ele in data_type:
            self.assertEqual(self.compute_base(self.x1,self.modlist[type_ele]),self.compute_mont(self.x1,self.modlist[type_ele]))

        
    def test_tensor_dtype(self):
        ##检查不同曲线转换的类型是否正确
        def my_function(in_a):
            try:
                torch.to_mod(in_a)
                return True
            except Exception as e:
                print(f"An error occurred: {e}")
                return False
        
        for type_ele in data_type:
            input = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=type_ele)
            result =my_function(input)
            self.assertTrue(result)

    def test_tensor_func(self):
        ###测试add,sub等函数的正确性
        def test_add(self):
            x1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.uint64)
            x2 =torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.uint64)
            ###TODO 底层实现
            sum_of_base=F.add_base(x1,x2)
            self.assertCustomEqual(F.add_mod(x1,x2),F.to_mod(sum_of_base))

        def test_sub(self):
            x1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.uint64)
            x2 =torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.uint64)
            ###TODO 底层实现
            sum_of_base=F.sub_base(x1,x2)
            self.assertCustomEqual(F.sub_mod(x1,x2),F.to_mod(sum_of_base))
        
        def test_mul(self):
            x1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)
            x2 =torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)
            ###TODO 底层实现
            sum_of_base=torch.mul_base(x1,x2)
            self.assertCustomEqual(F.mul_mod(x1,x2),F.to_mod(sum_of_base))

        test_mul(self)
              





if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CustomTestCase))
    runner = unittest.TextTestRunner()
    runner.run(suite)
