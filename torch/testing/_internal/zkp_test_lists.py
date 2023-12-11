import torch.nn.functional as F
import torch
import torch
import torch.nn.functional as F
import os
import unittest
import torchviz
import random
'''
测试的主要内容：
1.测试mont的转换正确性 
2.测试add等运算符的运算的正确性
3.验证不同曲线的正确性
'''
data_type=[torch.BLS12_381_Fr_G1_Base,torch.BLS12_377_Fr_G1_Base]

class CustomTestCase(unittest.TestCase):

    def setUp(self):
        self.x1 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_381_Fr_G1_Base)
        self.x2 =torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_381_Fr_G1_Base)
        self.x=[self.x1,self.x2]
        comput_mod_func = lambda mod1, mod2, mod3, mod4: mod1 + (mod2 << 64) + (mod3 << 128) + (mod4 <<192)
        self.modlist={
    torch.BLS12_381_Fr_G1_Base: comput_mod_func(0xffffffff00000001,0x53bda402fffe5bfe,0x3339d80809a1d805,0x73eda753299d7d48),
    torch.BLS12_377_Fr_G1_Base: comput_mod_func(0x0a11800000000001,0x59aa76fed0000001,0x60b44d1e5c37b001,0x12ab655e9a2ca556)
}

    def compute_base(self,in_a,mod):
        in_a=in_a.tolist()
        rows, cols =len(in_a), len(in_a[0])
        for i in range(1):
            res=0
            for j in range(cols):
                res+=(int(in_a[i][j]))*(2**(j*64))%mod
                # if(j==3):
            res=(res*(2**256))%mod
        return res
    
    def compute_mont(self,in_a):
        in_a=in_a.tolist()
        rows, cols =len(in_a), len(in_a[0])
        for i in range(1):
            res=0
            for j in range(cols):
                res+=(int(in_a[i][j]))*(2**(j*64))
        return res

    def assertCustomEqual(self, first, second, msg=None):
         if not torch.equal_mod(first, second):
            raise ValueError("Tensors are not equal")
        # self.assertEqual(first, second, msg=msg)

    def test_custom_data_equality(self):
        #蒙哥马利域下的比较
        custom_data_1 =F.to_mont(self.x1)
        custom_data_2 =F.to_mont(self.x1)
        # print(custom_data_1,custom_data_2)
        self.assertCustomEqual(custom_data_1, custom_data_2, "自定义数据应该相等")

    def test_tensor_dtype(self):
        ##检查不同曲线转换的类型是否正确
        def my_function(in_a):
            try:
                torch.to_mont(in_a)
                return True
            except Exception as e:
                print(f"An error occurred: {e}")
                return False
        
        for type_ele in data_type:
            input = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=type_ele)
            result =my_function(input)
            self.assertTrue(result)

    def test_custom_data_type(self):
        #检查mont转换的结果是否正确，和直接计算的结果比对
        def simple_type_test(in_a):
            try:
                custom_data_1=F.to_mont(self.x1)
                test_to_base_value=F.to_base(custom_data_1)
                self.assertEqual(test_to_base_value.dtype, torch.BLS12_381_Fr_G1_Base )
            except Exception as e:
                print(f"An error occurred: {e}")
                return False

        for index in range(len(data_type)):
            self.assertEqual(self.compute_base(self.x[index],self.modlist[self.x[index].dtype]),self.compute_mont(F.to_mont(self.x[index])))

    def test_add(self):
            #####生成测试数据
            min_dimension = 4  # 最低维度为4

            # 随机确定行数和列数（至少4行4列）
            rows = random.randint(min_dimension, min_dimension)  # 至少4行，最多7行
            columns = random.randint(min_dimension, min_dimension)  # 至少4列，最多7列


            for type_element in data_type:
                # 生成随机整数二维数组
                random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                random_array_2 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                t1 = torch.tensor(random_array_1,dtype=type_element)
                t2 = torch.tensor(random_array_2,dtype=type_element)
                # t1=torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)

                # t2=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)
                #################################################################
                base1=self.compute_base(t1,self.modlist[type_element])
                base2=self.compute_base(t2,self.modlist[type_element])

                r1=F.to_mont(t1)
                r2=F.to_mont(t2)

                mont1=self.compute_mont(r1)
                mont2=self.compute_mont(r2)
                # print(mont1,mont2)
                # print(self.modlist[type_element])
                self.assertEqual((mont1+mont2)%self.modlist[type_element],self.compute_mont(F.add_mod(r1,r2)))

    def test_sub(self):
            #####生成测试数据
            min_dimension = 4  # 最低维度为4

            # 随机确定行数和列数（至少4行4列）
            rows = random.randint(min_dimension, min_dimension)  # 至少4行，最多7行
            columns = random.randint(min_dimension, min_dimension)  # 至少4列，最多7列


            for type_element in data_type:
                # 生成随机整数二维数组
                random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                random_array_2 = [[random.randint(0, random_array_1[i][j]) for j in range(columns)]for i in range(rows)]

                t1 = torch.tensor(random_array_1,dtype=type_element)
                t2 = torch.tensor(random_array_2,dtype=type_element)
                #################################################################
                base1=self.compute_base(t1,self.modlist[type_element])
                base2=self.compute_base(t2,self.modlist[type_element])

                r1=F.to_mont(t1)
                r2=F.to_mont(t2)

                mont1=self.compute_mont(r1)
                mont2=self.compute_mont(r2)
                self.assertEqual((mont1-mont2)%self.modlist[type_element],self.compute_mont(F.sub_mod(r1,r2)))

    def test_mul(self):
        #####生成测试数据
            min_dimension = 4  # 最低维度为4

            # 随机确定行数和列数（至少4行4列）
            rows = random.randint(min_dimension, min_dimension)  # 至少4行，最多7行
            columns = random.randint(min_dimension, min_dimension)  # 至少4列，最多7列


            for type_element in data_type:
                # 生成随机整数二维数组
                random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                random_array_2 =[[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]

                t1 = torch.tensor(random_array_1,dtype=type_element)
                t2 = torch.tensor(random_array_2,dtype=type_element)
                #################################################################
                base1=self.compute_base(t1,self.modlist[type_element])
                base2=self.compute_base(t2,self.modlist[type_element])

                r1=F.to_mont(t1)
                r2=F.to_mont(t2)

                mont1=self.compute_mont(r1)
                mont2=self.compute_mont(r2)
                self.assertEqual(((mont1*mont2))%self.modlist[type_element],((self.compute_mont(F.mul_mod(r1,r2)))<<256) %self.modlist[type_element])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CustomTestCase))
    runner = unittest.TextTestRunner()
    runner.run(suite)
