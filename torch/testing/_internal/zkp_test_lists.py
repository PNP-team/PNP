import torch.nn.functional as F
import torch
import torch
import torch.nn.functional as F
import os
import unittest
import torchviz
import random
'''
Main contents of the test:
1. Test the correctness of mont's conversion
2. Test the correctness of operations of operators such as add
3. Verify the correctness of different curves
'''
data_type=[torch.BLS12_381_Fr_G1_Base,torch.BLS12_377_Fr_G1_Base]

class CustomTestCase(unittest.TestCase):

    def setUp(self):
        self.x1 = torch.tensor([[1, 2, 3, 2], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_381_Fr_G1_Base)
        self.x2 =torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8],[1,2,3,4]], dtype=torch.BLS12_381_Fr_G1_Base)
        self.x=[self.x1,self.x2]
        self.p={torch.BLS12_377_Fr_G1_Base:0x01ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001
                ,torch.BLS12_381_Fr_G1_Base:0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab}
        comput_mod_func = lambda mod1, mod2, mod3, mod4: mod1 + (mod2 << 64) + (mod3 << 128) + (mod4 <<192)
        self.modlist={
    torch.BLS12_381_Fr_G1_Base: comput_mod_func(0xffffffff00000001,0x53bda402fffe5bfe,0x3339d80809a1d805,0x73eda753299d7d48),
    torch.BLS12_377_Fr_G1_Base: comput_mod_func(0x0a11800000000001,0x59aa76fed0000001,0x60b44d1e5c37b001,0x12ab655e9a2ca556)
}

    def compute_base_mod(self,in_a,mod):
        in_a=in_a.tolist()
        rows, cols =len(in_a), len(in_a[0])
        for i in range(1):
            res=0
            for j in range(cols):
                res+=(int(in_a[i][j]))*(2**(j*64))%mod
                # if(j==3):
            res=(res*(2**256))%mod
        return res
    
    def compute_base(self,in_a):
        in_a=in_a.tolist()
        rows, cols =len(in_a), len(in_a[0])
        for i in range(1):
            res=0
            for j in range(cols):
                res+=(int(in_a[i][j]))*(2**(j*64))
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
        custom_data_1 =F.to_mont(self.x1)
        custom_data_2 =F.to_mont(self.x1)
        # print(custom_data_1,custom_data_2)
        self.assertCustomEqual(custom_data_1, custom_data_2, "Custom data should be equal")

    def test_tensor_dtype(self):
        ##Check if the different curve transformations are of the correct type
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
        #Check whether the result of mont conversion is correct and compare it with the result of direct calculation
        def simple_type_test(in_a):
            try:
                custom_data_1=F.to_mont(self.x1)
                test_to_base_value=F.to_base(custom_data_1)



                self.assertEqual(test_to_base_value.dtype, torch.BLS12_381_Fr_G1_Base )
            except Exception as e:
                print(f"An error occurred: {e}")
                return False

        for index in range(len(data_type)):
            self.assertEqual(self.compute_base_mod(self.x[index],self.modlist[self.x[index].dtype]),self.compute_mont(F.to_mont(self.x[index])))

    def test_add(self):
            #####Generate test data
            min_dimension = 4  # The lowest dimension is 4

            #Randomly determine the number of rows and columns (at least 4 rows and 4 columns)
            rows = random.randint(min_dimension, min_dimension)  
            columns = random.randint(min_dimension, min_dimension)  


            for type_element in data_type:
                # Generate a two-dimensional array of random integers

                t1=[]
                t2=[]   ####check value < modulo
                while True:
                        random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                        random_array_2 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                        
                        t1 = torch.tensor(random_array_1, dtype=type_element)
                        t2 = torch.tensor(random_array_2, dtype=type_element)
                        
                        base1 = self.compute_base(t1)
                        base2 = self.compute_base(t2)
                        
                        if base1 < self.p[type_element] and base2 < self.p[type_element]:
                            break  # Exit loop if conditions are met

                    # t1=torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)

                    # t2=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)
                    #################################################################
                base1=self.compute_base_mod(t1,self.modlist[type_element])
                base2=self.compute_base_mod(t2,self.modlist[type_element])

                r1=F.to_mont(t1)
                r2=F.to_mont(t2)
                self.assertEqual((base1+base2)%self.modlist[type_element],self.compute_mont(F.add_mod(r1,r2)))

    def test_sub(self):

            min_dimension = 4  

            rows = random.randint(min_dimension, min_dimension) 
            columns = random.randint(min_dimension, min_dimension)  


            for type_element in data_type:
               
                t1=[]
                t2=[]   ####check value < modulo
                while True:
                        random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                        random_array_2 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                        
                        t1 = torch.tensor(random_array_1, dtype=type_element)
                        t2 = torch.tensor(random_array_2, dtype=type_element)
                        
                        base1 = self.compute_base(t1)
                        base2 = self.compute_base(t2)
                        
                        if base1 < self.p[type_element] and base2 < self.p[type_element]:
                            break  # Exit loop if conditions are met
                #################################################################
                base1=self.compute_base_mod(t1,self.modlist[type_element])
                base2=self.compute_base_mod(t2,self.modlist[type_element])

                r1=F.to_mont(t1)
                r2=F.to_mont(t2)

                self.assertEqual((base1-base2)%self.modlist[type_element],self.compute_mont(F.sub_mod(r1,r2)))

    def test_mul(self):
            min_dimension = 4  

  
            rows = random.randint(min_dimension, min_dimension)  
            columns = random.randint(min_dimension, min_dimension) 


            for type_element in data_type:
                t1=[]
                t2=[]   ####check value < modulo
                while True:
                        random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                        random_array_2 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
                        
                        t1 = torch.tensor(random_array_1, dtype=type_element)
                        t2 = torch.tensor(random_array_2, dtype=type_element)
                        
                        base1 = self.compute_base(t1)
                        base2 = self.compute_base(t2)
                        
                        if base1 < self.p[type_element] and base2 < self.p[type_element]:
                            break  # Exit loop if conditions are met
                t2 = torch.tensor(random_array_2,dtype=type_element)
                #################################################################
                base1=self.compute_base(t1)
                base2=self.compute_base(t2)
                r1=F.to_mont(t1)
                r2=F.to_mont(t2)
                self.assertEqual(((base1*base2)<<256)%self.modlist[type_element],((self.compute_mont(F.mul_mod(r1,r2)))) %self.modlist[type_element])

    def test_special(self):
        #####Generate test data
        min_dimension = 5  

        #Randomly determine the number of rows and columns 
        rows = random.randint(min_dimension, min_dimension)  
        columns = random.randint(min_dimension, min_dimension)  
        for type_element in data_type:

            random_array_1 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]
            random_array_2 = [[random.randint(0, 2**64) for _ in range(columns)] for _ in range(rows)]

            t1 = torch.tensor(random_array_1,dtype=type_element)
            F.to_mont(t1)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CustomTestCase))
    runner = unittest.TextTestRunner()
    runner.run(suite)
